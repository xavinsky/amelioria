"""Construction des données et des trainers de préférences, partagée par run.py et la variante itérative."""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(next(p for p in pathlib.Path(__file__).resolve().parents if (p / "commun").is_dir())))

import torch
from datasets import Dataset

from commun import donnees, entrainement, modeles

# Pertes « paires » disponibles dans DPOTrainer (même données, même trainer, seule la perte change)
PERTES_DPO = {
    "dpo": "sigmoid", "dpo_norm": "sigmoid_norm", "ipo": "ipo", "rdpo": "robust", "slic": "hinge", "apo_zero": "apo_zero", "apo_down": "apo_down",
    "nca": "nca_pair", "bco": "bco_pair", "sppo": "sppo_hard", "exo": "exo_pair", "aot": "aot", "discopop": "discopop",
}
ALGOS = sorted(PERTES_DPO) + ["orpo", "simpo", "cpo", "kto"]


def paire(tok, question: str, chosen: str, rejected: str) -> dict:
    """Paire au format TRL (conversationnel si chat template, texte sinon)."""
    ex = modeles.exemple_sft(tok, question, chosen)
    rej = modeles.exemple_sft(tok, question, rejected)["completion"]
    return {"prompt": ex["prompt"], "chosen": ex["completion"], "rejected": rej}


def charger_paires(cfg, tok, split: str, n: int) -> list[dict]:
    d = cfg.donnees
    if d.source == "arith":
        graine = d.graine + (0 if split == "train" else 10_000)
        return [paire(tok, p["question"], p["chosen"], p["rejected"]) for p in donnees.preferences_arith(n, graine, d.niveau)]
    if d.source == "hf":
        # ex. trl-lib/ultrafeedback_binarized : chosen/rejected = conversations complètes (prompt implicite)
        return donnees.preferences_hf(d.nom, d[f"split_{split}"], n, d.graine)
    raise ValueError(d.source)


def _decomposer(ex: dict):
    """(prompt texte, réponse préférée, réponse rejetée) pour le calcul de log-probabilités."""
    if "prompt" in ex:
        prompt, chosen, rejected = ex["prompt"], ex["chosen"], ex["rejected"]
    else:
        prompt, chosen, rejected = ex["chosen"][:-1], ex["chosen"][-1:], ex["rejected"][-1:]
    contenu = lambda c: c[0]["content"] if isinstance(c, list) else c
    return prompt, contenu(chosen), contenu(rejected)


@torch.no_grad()
def exactitude_preferences(model, tok, paires: list[dict]) -> dict:
    """Part des paires où le modèle donne une log-prob moyenne par token plus haute à la réponse préférée."""
    ok, marges = 0, []
    for ex in paires:
        prompt, chosen, rejected = _decomposer(ex)
        if isinstance(prompt, list):
            prompt = tok.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        lp = modeles.logprob_suites(model, tok, [prompt, prompt], [chosen, rejected])
        longueurs = [max(len(tok(t, add_special_tokens=False)["input_ids"]), 1) for t in (chosen, rejected)]
        marge = (lp[0] / longueurs[0] - lp[1] / longueurs[1]).item()
        marges.append(marge)
        ok += marge > 0
    return {"exactitude_preferences": round(ok / len(paires), 4), "marge_moyenne": round(sum(marges) / len(marges), 4)}


def construire_trainer(cfg, model, tok, paires: list[dict], sortie: str):
    algo = cfg.algo
    e = cfg.entrainement
    peft_config = entrainement.config_lora(cfg.peft) if cfg.peft.get("activer") else None
    commun = entrainement.arguments(e, sortie, max_length=e.max_length, beta=cfg.beta)
    ds = Dataset.from_list(paires)

    if algo in PERTES_DPO:
        from trl import DPOConfig, DPOTrainer

        lissage = cfg.get("label_smoothing") or 0.0
        if lissage == 0.0 and algo in ("exo", "rdpo"):
            lissage = {"exo": 1e-3, "rdpo": 0.1}[algo]   # EXO exige un lissage > 0 ; rDPO à 0 équivaut à DPO
        pertes, poids = [PERTES_DPO[algo]], [1.0]
        if cfg.get("poids_sft"):   # terme SFT sur la réponse préférée (RPO) : empêche sa vraisemblance de s'effondrer
            pertes, poids = pertes + ["sft"], poids + [cfg.poids_sft]
        args = DPOConfig(**commun, loss_type=pertes, loss_weights=poids, label_smoothing=lissage)
        return DPOTrainer(model=model, args=args, train_dataset=ds, processing_class=tok, peft_config=peft_config)
    if algo == "orpo":
        from trl.experimental.orpo import ORPOConfig, ORPOTrainer

        return ORPOTrainer(model=model, args=ORPOConfig(**commun), train_dataset=ds, processing_class=tok, peft_config=peft_config)
    if algo in ("simpo", "cpo"):
        from trl.experimental.cpo import CPOConfig, CPOTrainer

        extra = {"loss_type": "simpo", "cpo_alpha": 0.0, "simpo_gamma": cfg.get("simpo_gamma", 0.5)} if algo == "simpo" else {}
        return CPOTrainer(model=model, args=CPOConfig(**commun, **extra), train_dataset=ds, processing_class=tok, peft_config=peft_config)
    if algo == "kto":
        from trl import KTOConfig, KTOTrainer

        # KTO n'a pas besoin de paires : chaque réponse est simplement étiquetée bonne ou mauvaise.
        non_appariees = []
        for ex in paires:
            prompt, chosen, rejected = (ex["prompt"], ex["chosen"], ex["rejected"]) if "prompt" in ex else (ex["chosen"][:-1], ex["chosen"][-1:], ex["rejected"][-1:])
            non_appariees += [{"prompt": prompt, "completion": chosen, "label": True},
                              {"prompt": prompt, "completion": rejected, "label": False}]
        return KTOTrainer(model=model, args=KTOConfig(**commun), train_dataset=Dataset.from_list(non_appariees),
                          processing_class=tok, peft_config=peft_config)
    raise ValueError(f"algo inconnu : {algo} (disponibles : {ALGOS})")


def prompts_test(cfg, n: int, longueur_max: int = 2000) -> list[str]:
    """Demandes du split de test du jeu de préférences (premier message de l'utilisateur), jamais vues à
    l'entraînement ; les très longues sont écartées pour borner la génération."""
    from datasets import load_dataset

    d = cfg.donnees
    ds = load_dataset(d.nom, split=d.split_test).shuffle(seed=d.graine)
    prompts = [ex["chosen"][0]["content"] for ex in ds if len(ex["chosen"][0]["content"]) <= longueur_max]
    return prompts[:n]


def repondre(model, tok, prompts: list[str], ev) -> list[str]:
    """Réponses en décodage glouton : la même demande donne la même réponse, la comparaison est appariée."""
    textes = [modeles.prompt_texte(tok, p) for p in prompts]
    return [r.strip() for r in modeles.generer(model, tok, textes, max_new_tokens=ev.max_new_tokens, batch=ev.get("batch", 32))]


@torch.no_grad()
def noter(j, prompts: list[str], jeux: dict[str, list[str]]) -> dict[str, list[float]]:
    """Note de chaque (demande, réponse) par un modèle de récompense externe (section `juge` de la config).

    Le juge n'a vu ni le modèle entraîné ni ses sorties : il départage deux réponses à la même demande."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(j.nom)
    juge = AutoModelForSequenceClassification.from_pretrained(j.nom, dtype=torch.bfloat16, num_labels=1).to(modeles.appareil()).eval()
    notes = {}
    for nom, reponses in jeux.items():
        textes = [tok.apply_chat_template([{"role": "user", "content": p}, {"role": "assistant", "content": r}], tokenize=False)
                  for p, r in zip(prompts, reponses)]
        res = []
        for i in range(0, len(textes), j.get("batch", 8)):
            enc = tok(textes[i : i + j.get("batch", 8)], return_tensors="pt", padding=True, truncation=True,
                      max_length=j.get("max_length", 2048), add_special_tokens=False).to(juge.device)
            res += juge(**enc).logits[:, 0].float().tolist()
        notes[nom] = res
    del juge
    torch.cuda.empty_cache()
    return notes


def victoires(notes_a: list[float], notes_b: list[float]) -> dict:
    """Taux de victoire de b contre a (note du juge strictement plus haute) : 0,5 = aussi bien l'un que l'autre."""
    from commun import preuve

    return preuve.taux([b > a for a, b in zip(notes_a, notes_b)], "taux_victoire")
