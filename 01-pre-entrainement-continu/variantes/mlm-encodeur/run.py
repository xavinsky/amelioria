"""Pré-entraînement continu d'un ENCODEUR (BERT, RoBERTa, DeBERTa...) par masked language modeling.

Même idée que pour un LLM (DAPT, « Don't Stop Pretraining ») : on poursuit l'objectif d'origine
(ici prédire 15 % de tokens masqués) sur le corpus du domaine, avant de fine-tuner pour la tâche finale.
"""

import math
import pathlib
import sys

sys.path.insert(0, str(next(p for p in pathlib.Path(__file__).resolve().parents if (p / "commun").is_dir())))

from datasets import Dataset
from transformers import AutoModelForMaskedLM, AutoTokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments

from commun import config, donnees, entrainement, modeles, rapport, reprise


def main():
    cfg = config.charger(__file__)
    tok = AutoTokenizer.from_pretrained(cfg.modele.nom)
    model = AutoModelForMaskedLM.from_pretrained(cfg.modele.nom).to(modeles.appareil())

    kb = donnees.faits_fictifs(cfg.donnees.n_entites, cfg.donnees.graine)
    docs = kb["documents"]
    train = [x["texte"] for i, x in enumerate(docs) if i % 4 != 3]
    test = [x["texte"] for i, x in enumerate(docs) if i % 4 == 3]

    def encoder(textes):
        return Dataset.from_dict(tok(textes, truncation=True, max_length=cfg.entrainement.longueur_max))

    collator = DataCollatorForLanguageModeling(tok, mlm=True, mlm_probability=cfg.entrainement.proba_masque, seed=cfg.graine)
    trainer = Trainer(
        model=model,
        args=TrainingArguments(**entrainement.arguments(cfg.entrainement, cfg.sortie, per_device_eval_batch_size=16)),
        train_dataset=encoder(train),
        eval_dataset=encoder(test),
        data_collator=collator,
    )
    perte_avant = trainer.evaluate()["eval_loss"]
    reprise.entrainer(trainer)
    perte_apres = trainer.evaluate()["eval_loss"]

    if cfg.get("sauvegarder", True):
        trainer.save_model(f"{cfg.sortie}/modele")
        tok.save_pretrained(f"{cfg.sortie}/modele")
    rapport.enregistrer(
        cfg,
        avant={"perte_mlm_domaine": round(perte_avant, 4), "pseudo_perplexite": round(math.exp(perte_avant), 2)},
        apres={"perte_mlm_domaine": round(perte_apres, 4), "pseudo_perplexite": round(math.exp(perte_apres), 2)},
    )


if __name__ == "__main__":
    main()
