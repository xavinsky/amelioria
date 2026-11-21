"""Pré-entraînement continu d'un ENCODEUR (BERT, RoBERTa, DeBERTa...) par masked language modeling.

Même idée que pour un LLM (DAPT, « Don't Stop Pretraining ») : on poursuit l'objectif d'origine
(ici prédire 15 % de tokens masqués) sur le corpus du domaine, avant de fine-tuner pour la tâche finale.

Deux protocoles, selon `donnees.source` :
- `banking77` (défaut) : les messages de banking77, sans leurs étiquettes, servent au masquage ; puis le modèle est
  fine-tuné à classer les 77 intentions avec quelques exemples étiquetés par intention, et comparé au même
  fine-tuning sans masquage préalable, sur les mêmes messages de test ;
- `fiches` : biographies fictives, mesure de la seule perte de masquage.
"""

import math
import pathlib
import random
import sys

sys.path.insert(0, str(next(p for p in pathlib.Path(__file__).resolve().parents if (p / "commun").is_dir())))

import numpy as np
from datasets import Dataset, load_dataset
from transformers import (AutoModelForMaskedLM, AutoModelForSequenceClassification, AutoTokenizer, DataCollatorForLanguageModeling,
                          DataCollatorWithPadding, Trainer, TrainingArguments, set_seed)

from commun import config, donnees, entrainement, modeles, preuve, rapport, reprise


def banking77(cfg):
    d = cfg.donnees
    ds = load_dataset(d.nom)
    tok = AutoTokenizer.from_pretrained(cfg.modele.nom)
    brut = ds["train"].shuffle(seed=cfg.graine).select(range(min(d.n_mlm, len(ds["train"]))))
    test = ds["test"].select(range(min(d.n_test, len(ds["test"]))))

    # Étiquetés : n_par_classe messages tirés par intention (selon la graine). Les mêmes messages, sans leurs
    # étiquettes, font partie du corpus de masquage, comme les autres messages d'entraînement.
    rng, par_classe = random.Random(cfg.graine), {}
    for i, etiquette in enumerate(brut[d.colonne_label]):
        par_classe.setdefault(etiquette, []).append(i)
    etiquetes = brut.select(sorted(i for e in sorted(par_classe) for i in rng.sample(par_classe[e], min(d.n_par_classe, len(par_classe[e])))))
    n_classes = ds["train"].features[d.colonne_label].num_classes

    def encoder(lot):
        return tok(lot[d.colonne_texte], truncation=True, max_length=cfg.entrainement.longueur_max)

    def masquage():
        textes = brut[d.colonne_texte]
        a_garder = [t for i, t in enumerate(textes) if i % 20 != 0]
        a_tester = [t for i, t in enumerate(textes) if i % 20 == 0]
        model = AutoModelForMaskedLM.from_pretrained(cfg.modele.nom).to(modeles.appareil())
        collator = DataCollatorForLanguageModeling(tok, mlm=True, mlm_probability=cfg.entrainement.proba_masque, seed=cfg.graine)
        enc = lambda x: Dataset.from_dict(tok(x, truncation=True, max_length=cfg.entrainement.longueur_max))  # noqa: E731
        trainer = Trainer(model=model, args=TrainingArguments(**entrainement.arguments(cfg.entrainement, f"{cfg.sortie}/masquage", per_device_eval_batch_size=128)),
                          train_dataset=enc(a_garder), eval_dataset=enc(a_tester), data_collator=collator)
        avant = trainer.evaluate()["eval_loss"]
        reprise.entrainer(trainer)
        apres = trainer.evaluate()["eval_loss"]
        trainer.save_model(f"{cfg.sortie}/masquage/modele")
        tok.save_pretrained(f"{cfg.sortie}/masquage/modele")
        entrainement.liberer(trainer)
        return {"perte_mlm_avant": round(avant, 4), "perte_mlm_apres": round(apres, 4),
                "messages_masquage": len(a_garder)}

    def classer(depart, nom):
        set_seed(cfg.graine)
        model = AutoModelForSequenceClassification.from_pretrained(depart, num_labels=n_classes).to(modeles.appareil())
        colonnes = lambda x: x.map(encoder, batched=True).rename_column(d.colonne_label, "labels")  # noqa: E731
        dernier = {}

        def metriques(pred):
            dernier["succes"] = (np.argmax(pred.predictions, -1) == pred.label_ids).tolist()
            return {"exactitude": float(np.mean(dernier["succes"]))}

        trainer = Trainer(model=model, args=TrainingArguments(**entrainement.arguments(cfg.classification, f"{cfg.sortie}/{nom}", per_device_eval_batch_size=128)),
                          train_dataset=colonnes(etiquetes), eval_dataset=colonnes(test),
                          data_collator=DataCollatorWithPadding(tok), compute_metrics=metriques)
        reprise.entrainer(trainer)
        trainer.evaluate()
        entrainement.liberer(trainer)
        return preuve.taux(dernier["succes"])

    mlm = reprise.etape("masquage", masquage)
    sans = reprise.etape("temoin", lambda: classer(cfg.modele.nom, "temoin"))
    avec = reprise.etape("avec_masquage", lambda: classer(f"{cfg.sortie}/masquage/modele", "avec_masquage"))
    if not cfg.get("sauvegarder", True):
        import shutil

        shutil.rmtree(f"{cfg.sortie}/masquage/modele", ignore_errors=True)
    rapport.enregistrer(cfg, temoin_sans_masquage=sans, avec_masquage=avec, masquage={**mlm, "etiquetes": len(etiquetes)})


def fiches(cfg):
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


def main():
    cfg = config.charger(__file__)
    (banking77 if cfg.donnees.get("source", "banking77") == "banking77" else fiches)(cfg)


if __name__ == "__main__":
    main()
