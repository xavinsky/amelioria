"""Briques d'entraînement partagées (arguments communs, SFT, LoRA)."""

from __future__ import annotations

import contextlib

from datasets import Dataset

from . import config, modeles, reprise

# Clés courtes de la section `entrainement` des configs -> champs de transformers.TrainingArguments
_CORRESPONDANCE = {
    "lr": "learning_rate",
    "epochs": "num_train_epochs",
    "max_steps": "max_steps",
    "batch": "per_device_train_batch_size",
    "accumulation": "gradient_accumulation_steps",
    "warmup": "warmup_steps",
    "scheduler": "lr_scheduler_type",
    "weight_decay": "weight_decay",
    "optim": "optim",
    "gradient_checkpointing": "gradient_checkpointing",
    "logging_steps": "logging_steps",
    "max_grad_norm": "max_grad_norm",
}


def arguments(e, sortie: str, **extra) -> dict:
    """Arguments TrainingArguments communs à tous les trainers (SFT, DPO, GRPO, Reward...)."""
    args = {
        "output_dir": sortie,
        "report_to": "none",
        "save_strategy": "no",
        "logging_steps": 5,
        "bf16": modeles.bf16_actif(),
        "seed": config.GRAINE,
        "remove_unused_columns": True,
        "gradient_checkpointing": False,   # les configs TRL l'activent par défaut : seule la clé de config décide ici
    }
    for court, long in _CORRESPONDANCE.items():
        if e.get(court) is not None:
            args[long] = e[court]
    for k, v in (e.get("autres") or {}).items():
        args[k] = v
    args.update(extra)
    return args


def liberer(trainer):
    """Libère les états de l'optimiseur après l'entraînement : un second entraînement ou une génération peut suivre.

    Le trainer reste utilisable pour `trainer.model` et `trainer.save_model()`.
    """
    import gc

    import torch

    trainer.optimizer = trainer.lr_scheduler = None
    if getattr(trainer, "ref_model", None) is not None:   # DPO, KTO... : modèle de référence
        trainer.ref_model = None
    if getattr(trainer, "callback_handler", None) is not None:
        trainer.callback_handler.optimizer = trainer.callback_handler.lr_scheduler = None
    acc = getattr(trainer, "accelerator", None)
    if acc is not None:
        acc._optimizers.clear()
        acc._schedulers.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return trainer


def config_lora(p):
    """Section `peft` d'une config -> LoraConfig (utilisé par plusieurs méthodes)."""
    from peft import LoraConfig

    return LoraConfig(
        r=p.get("r", 16),
        lora_alpha=p.get("alpha", 32),
        lora_dropout=p.get("dropout", 0.05),
        target_modules=p.get("modules", "all-linear"),
        use_dora=p.get("dora", False),
        use_rslora=p.get("rslora", False),
        init_lora_weights=p.get("init", True),
        task_type="CAUSAL_LM",
    )


def sft(model, tok, exemples: list[dict], e, sortie: str, peft_config=None, max_length: int = 512):
    """Fine-tuning supervisé TRL sur des exemples prompt/completion (voir modeles.exemple_sft).

    Renvoie le trainer : `trainer.model` est le modèle entraîné, `trainer.save_model()` sauvegarde
    correctement y compris en FSDP / DeepSpeed ZeRO-3.
    """
    from trl import SFTConfig, SFTTrainer

    conf = SFTConfig(**arguments(e, sortie, max_length=max_length, completion_only_loss=True))
    trainer = SFTTrainer(model=model, args=conf, train_dataset=Dataset.from_list(exemples),
                         processing_class=tok, peft_config=peft_config)
    with mesurer() as mesures:
        reprise.entrainer(trainer)
    trainer.mesures = mesures
    return liberer(trainer)


@contextlib.contextmanager
def mesurer():
    """Durée et pic de mémoire GPU d'un entraînement : le coût de la méthode, à côté de son gain."""
    import time

    import torch

    mesures = {}
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    yield mesures
    mesures["duree_min"] = round((time.time() - t0) / 60, 1)
    if torch.cuda.is_available():
        mesures["memoire_gpu_max_gio"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)


def blocs_texte(tok, textes: list[str], longueur_bloc: int) -> Dataset:
    """Tokenise, concatène (séparés par EOS) et découpe en blocs de taille fixe : format du pré-entraînement."""
    ids: list[int] = []
    for t in textes:
        ids.extend(tok(t, add_special_tokens=False)["input_ids"] + [tok.eos_token_id])
    n = max(len(ids) // longueur_bloc, 1)
    blocs = [ids[i * longueur_bloc : (i + 1) * longueur_bloc] for i in range(n)]
    blocs = [b for b in blocs if len(b) == longueur_bloc] or [ids[:longueur_bloc]]
    return Dataset.from_dict({"input_ids": blocs})


def lm_causal(model, tok, textes: list[str], e, sortie: str, longueur_bloc: int = 512, callbacks=None, **extra):
    """Entraînement « prochain token » sur du texte brut (pré-entraînement continu, cicatrisation...)."""
    from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments

    ds = blocs_texte(tok, textes, longueur_bloc)
    print(f"[entrainement] {len(ds)} blocs de {longueur_bloc} tokens")
    trainer = Trainer(
        model=model,
        args=TrainingArguments(**arguments(e, sortie, **extra)),
        train_dataset=ds,
        data_collator=DataCollatorForLanguageModeling(tok, mlm=False),
        callbacks=callbacks,
    )
    reprise.entrainer(trainer)
    return liberer(trainer)


def tokeniser_prompt_reponse(tok, exemples: list[dict], max_length: int = 512, cle_reponse: str = "reponse") -> list[dict]:
    """Pour les boucles d'entraînement écrites à la main : input_ids + labels masqués sur le prompt."""
    lignes = []
    for ex in exemples:
        prompt = modeles.prompt_texte(tok, ex["question"])
        ids_p = tok(prompt, add_special_tokens=not modeles.a_chat_template(tok))["input_ids"]
        ids_r = tok(ex[cle_reponse], add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
        ids = (ids_p + ids_r)[:max_length]
        lignes.append({"input_ids": ids, "labels": ([-100] * len(ids_p) + ids_r)[:max_length]})
    return lignes


def collate_causal(tok):
    """Padding à droite d'exemples {input_ids, labels} -> tenseurs input_ids, attention_mask, labels."""
    import torch

    def f(lot):
        n = max(len(x["input_ids"]) for x in lot)
        ids = torch.full((len(lot), n), tok.pad_token_id)
        lab = torch.full((len(lot), n), -100)
        att = torch.zeros((len(lot), n), dtype=torch.long)
        for i, x in enumerate(lot):
            k = len(x["input_ids"])
            ids[i, :k], lab[i, :k], att[i, :k] = torch.tensor(x["input_ids"]), torch.tensor(x["labels"]), 1
        return {"input_ids": ids, "attention_mask": att, "labels": lab}

    return f


def logprob_sequences(model, lot):
    """(somme des log-probabilités des tokens de réponse, nombre de ces tokens), par séquence."""
    import torch

    logits = model(input_ids=lot["input_ids"], attention_mask=lot["attention_mask"]).logits[:, :-1].float()
    labels = lot["labels"][:, 1:]
    masque = labels != -100
    lp = torch.log_softmax(logits, -1).gather(-1, labels.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    return (lp * masque).sum(-1), masque.sum(-1)


def exemples_sft(tok, qa: list[dict], systeme: str | None = None) -> list[dict]:
    return [modeles.exemple_sft(tok, ex.get("question"), ex["reponse"], ex.get("systeme", systeme)) for ex in qa]


def remelange(model, tok, r, n_tache: int, cache_dir) -> list[dict]:
    """Exemples de remélange contre l'oubli : des demandes générales et les réponses du modèle AVANT entraînement.

    Mêlés aux exemples de la tâche (part `r.part`), ils apprennent au modèle à garder sa façon de répondre hors de
    la tâche : sans eux, un SFT sur des réponses d'un mot rend le modèle laconique partout.
    Renvoie des exemples au format commun, sans message système (`systeme: None`).
    """
    n = round(r.get("part", 0.15) * n_tache / (1 - r.get("part", 0.15)))
    return reponses_du_modele(model, tok, n, cache_dir, r.get("max_new_tokens", 512), r.get("batch", 32), r.get("graine", 0))


def reponses_du_modele(model, tok, n: int, cache_dir, max_new_tokens: int = 512, batch: int = 32, graine: int = 0) -> list[dict]:
    """n demandes générales (Alpaca) et les réponses complètes du modèle, générées une fois puis gardées en cache
    (même modèle, mêmes demandes -> mêmes réponses). Sert au remélange et à la distillation sur des conversations."""
    import json
    import pathlib

    from . import donnees

    cache = pathlib.Path(cache_dir) / f"{model.config._name_or_path.replace('/', '--')}-{n}-complets.json"
    if cache.exists():
        return json.loads(cache.read_text())
    # Une réponse coupée par la limite de tokens apprendrait au modèle à s'arrêter en pleine phrase : on n'en garde
    # que des complètes, en générant pour plus de demandes qu'il n'en faut.
    consignes = donnees.consignes_generales(2 * n, graine)
    reponses = modeles.generer(model, tok, [modeles.prompt_texte(tok, c) for c in consignes],
                               max_new_tokens=max_new_tokens, batch=batch)
    exemples = [{"question": c, "reponse": rep.strip(), "systeme": None} for c, rep in zip(consignes, reponses)
                if len(tok(rep, add_special_tokens=False)["input_ids"]) < max_new_tokens - 1][:n]
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(exemples, ensure_ascii=False))
    return exemples
