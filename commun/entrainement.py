"""Briques d'entraînement partagées (arguments communs, SFT, LoRA)."""

from __future__ import annotations

from datasets import Dataset

from . import modeles

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
        "seed": 42,
        "remove_unused_columns": True,
    }
    for court, long in _CORRESPONDANCE.items():
        if e.get(court) is not None:
            args[long] = e[court]
    for k, v in (e.get("autres") or {}).items():
        args[k] = v
    args.update(extra)
    return args


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
    trainer.train()
    return trainer


def blocs_texte(tok, textes: list[str], longueur_bloc: int) -> Dataset:
    """Tokenise, concatène (séparés par EOS) et découpe en blocs de taille fixe : format du pré-entraînement."""
    ids: list[int] = []
    for t in textes:
        ids.extend(tok(t, add_special_tokens=False)["input_ids"] + [tok.eos_token_id])
    n = max(len(ids) // longueur_bloc, 1)
    blocs = [ids[i * longueur_bloc : (i + 1) * longueur_bloc] for i in range(n)]
    blocs = [b for b in blocs if len(b) == longueur_bloc] or [ids[:longueur_bloc]]
    return Dataset.from_dict({"input_ids": blocs})


def lm_causal(model, tok, textes: list[str], e, sortie: str, longueur_bloc: int = 512, **extra):
    """Entraînement « prochain token » sur du texte brut (pré-entraînement continu, cicatrisation...)."""
    from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments

    ds = blocs_texte(tok, textes, longueur_bloc)
    print(f"[entrainement] {len(ds)} blocs de {longueur_bloc} tokens")
    trainer = Trainer(
        model=model,
        args=TrainingArguments(**arguments(e, sortie, **extra)),
        train_dataset=ds,
        data_collator=DataCollatorForLanguageModeling(tok, mlm=False),
    )
    trainer.train()
    return trainer


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
    return [modeles.exemple_sft(tok, ex["question"], ex["reponse"], systeme) for ex in qa]
