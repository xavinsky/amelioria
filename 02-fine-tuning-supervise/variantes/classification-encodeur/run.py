"""Fine-tuning supervisé d'un ENCODEUR pour la classification de texte (sentiment, intention, modération...).

Le cas le plus fréquent en production hors LLM génératifs : une tête linéaire sur BERT / RoBERTa /
ModernBERT, entraînée avec le modèle entier.
"""

import pathlib
import sys

sys.path.insert(0, str(next(p for p in pathlib.Path(__file__).resolve().parents if (p / "commun").is_dir())))

import numpy as np
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments

from commun import config, entrainement, modeles, preuve, rapport, reprise


def main():
    cfg = config.charger(__file__)
    d = cfg.donnees
    ds = load_dataset(d.nom, d.get("config"))
    train = ds[d.split_train].shuffle(seed=cfg.graine).select(range(min(d.n_train, len(ds[d.split_train]))))
    test = ds[d.split_test].shuffle(seed=cfg.graine).select(range(min(d.n_test, len(ds[d.split_test]))))
    n_labels = len(set(train[d.colonne_label]))

    tok = AutoTokenizer.from_pretrained(cfg.modele.nom)
    model = AutoModelForSequenceClassification.from_pretrained(cfg.modele.nom, num_labels=n_labels).to(modeles.appareil())

    def encoder(lot):
        return tok(lot[d.colonne_texte], truncation=True, max_length=cfg.entrainement.max_length)

    train = train.map(encoder, batched=True).rename_column(d.colonne_label, "labels")
    test = test.map(encoder, batched=True).rename_column(d.colonne_label, "labels")

    dernier = {}   # détail par exemple de la dernière évaluation, pour l'intervalle et la comparaison appariée

    def metriques(pred):
        dernier["succes"] = (np.argmax(pred.predictions, -1) == pred.label_ids).tolist()
        return {"exactitude": float(np.mean(dernier["succes"]))}

    trainer = Trainer(
        model=model,
        args=TrainingArguments(**entrainement.arguments(cfg.entrainement, cfg.sortie, per_device_eval_batch_size=64)),
        train_dataset=train, eval_dataset=test,
        data_collator=DataCollatorWithPadding(tok), compute_metrics=metriques,
    )
    avant = trainer.evaluate()  # tête de classification aléatoire : ~ hasard
    avant_taux = preuve.taux(dernier["succes"])
    reprise.entrainer(trainer)
    apres = trainer.evaluate()

    if cfg.get("sauvegarder", True):
        trainer.save_model(f"{cfg.sortie}/modele")
        tok.save_pretrained(f"{cfg.sortie}/modele")
    rapport.enregistrer(cfg, avant={**avant_taux, "perte": round(avant["eval_loss"], 4)},
                        apres={**preuve.taux(dernier["succes"]), "perte": round(apres["eval_loss"], 4)})


if __name__ == "__main__":
    main()
