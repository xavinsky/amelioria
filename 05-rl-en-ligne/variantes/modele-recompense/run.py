"""Entraîner un modèle de récompense à partir de préférences (1re étape du RLHF).

Un LLM + une tête scalaire, entraîné avec la perte de Bradley-Terry : score(préférée) > score(rejetée).
Le modèle obtenu sert ensuite de récompense dans run.py (`recompense.source=modele`), pour du best-of-N
(dossier 20) ou pour classer des paires on-policy (DPO itératif, dossier 04).
"""

import pathlib
import sys

RACINE = next(p for p in pathlib.Path(__file__).resolve().parents if (p / "commun").is_dir())
sys.path.insert(0, str(RACINE))
sys.path.insert(0, str(RACINE / "04-optimisation-preferences"))

import torch  # noqa: E402
from datasets import Dataset  # noqa: E402
from transformers import AutoModelForSequenceClassification  # noqa: E402
from trl import RewardConfig, RewardTrainer  # noqa: E402

import preferences  # noqa: E402
from commun import config, entrainement, modeles, preuve, rapport  # noqa: E402


@torch.no_grad()
def exactitude(model, tok, paires):
    model.eval()
    ok = []
    for ex in paires:
        scores = []
        for cle in ("chosen", "rejected"):
            if isinstance(ex[cle], list):
                texte = tok.apply_chat_template(ex["prompt"] + ex[cle] if "prompt" in ex else ex[cle], tokenize=False)
            else:
                texte = ex["prompt"] + ex[cle]
            enc = tok(texte, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
            scores.append(model(**enc).logits[0, 0].item())
        ok.append(scores[0] > scores[1])
    return preuve.taux(ok, "exactitude_preferences")


def main():
    cfg = config.charger(__file__)
    tok = modeles.charger_tokenizer(cfg.modele.nom)
    dtype = modeles.dtype_calcul() if cfg.modele.get("dtype", "auto") == "auto" else getattr(torch, cfg.modele.dtype)
    model = AutoModelForSequenceClassification.from_pretrained(cfg.modele.nom, num_labels=1, dtype=dtype)
    model.config.pad_token_id = tok.pad_token_id
    model.to(modeles.appareil())

    paires = preferences.charger_paires(cfg, tok, "train", cfg.donnees.n_train)
    test = preferences.charger_paires(cfg, tok, "test", cfg.evaluation.n_paires)
    avant = exactitude(model, tok, test)  # tête aléatoire : ~50 %

    peft_config = None
    if cfg.peft.get("activer"):
        peft_config = entrainement.config_lora(cfg.peft)
        peft_config.task_type = "SEQ_CLS"
    args = RewardConfig(**entrainement.arguments(cfg.entrainement, cfg.sortie, max_length=cfg.entrainement.max_length,
                                                 center_rewards_coefficient=0.01))
    trainer = RewardTrainer(model=model, args=args, train_dataset=Dataset.from_list(paires), processing_class=tok, peft_config=peft_config)
    trainer.train()
    apres = exactitude(trainer.model, tok, test)

    if cfg.get("sauvegarder", True):
        trainer.save_model(f"{cfg.sortie}/modele")
        tok.save_pretrained(f"{cfg.sortie}/modele")
    rapport.enregistrer(cfg, avant=avant, apres=apres)


if __name__ == "__main__":
    main()
