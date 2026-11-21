"""Fine-tuning supervisé d'un modèle de VISION pour la classification d'images (ViT, DeiT, ConvNeXt, DINOv2, Swin...).

On remplace la tête de classification par une tête au bon nombre de classes et on entraîne tout le réseau.

Départ réaliste : une régression logistique sur les plongements (embeddings) figés du même modèle, mesurée sur les mêmes
images de test ; le fine-tuning doit la dépasser, pas seulement une tête de classification aléatoire.
"""

import pathlib
import sys

sys.path.insert(0, str(next(p for p in pathlib.Path(__file__).resolve().parents if (p / "commun").is_dir())))

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoImageProcessor, AutoModelForImageClassification, Trainer, TrainingArguments

from commun import config, entrainement, modeles, preuve, rapport, reprise


def main():
    cfg = config.charger(__file__)
    d = cfg.donnees
    ds = load_dataset(d.nom)
    train = ds[d.split_train].shuffle(seed=cfg.graine).select(range(d.n_train))
    test = ds[d.split_test].shuffle(seed=cfg.graine).select(range(d.n_test))
    noms_classes = train.features[d.colonne_label].names

    processeur = AutoImageProcessor.from_pretrained(cfg.modele.nom)
    model = AutoModelForImageClassification.from_pretrained(
        cfg.modele.nom, num_labels=len(noms_classes), ignore_mismatched_sizes=True,
        id2label=dict(enumerate(noms_classes)), label2id={n: i for i, n in enumerate(noms_classes)},
    ).to(modeles.appareil())

    def sonde_lineaire():
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from transformers import AutoModel

        encodeur = AutoModel.from_pretrained(cfg.modele.nom).to(modeles.appareil()).eval()

        def plongements(ds):
            res = []
            for i in range(0, len(ds), 128):
                images = [im.convert("RGB") for im in ds[i : i + 128][d.colonne_image]]
                pixels = processeur(images, return_tensors="pt")["pixel_values"].to(modeles.appareil())
                with torch.no_grad():
                    res.append(encodeur(pixel_values=pixels).last_hidden_state[:, 0].float().cpu().numpy())
            return np.concatenate(res)

        echelle = StandardScaler().fit(plongements(train))
        clf = LogisticRegression(max_iter=2000).fit(echelle.transform(plongements(train)), train[d.colonne_label])
        pred = clf.predict(echelle.transform(plongements(test)))
        del encodeur
        return preuve.taux([int(p) == int(y) for p, y in zip(pred, test[d.colonne_label])])

    base_sonde = sonde_lineaire()

    def collate(lot):
        images = [ex[d.colonne_image].convert("RGB") for ex in lot]
        pixels = processeur(images, return_tensors="pt")["pixel_values"]
        return {"pixel_values": pixels, "labels": torch.tensor([ex[d.colonne_label] for ex in lot])}

    dernier = {}   # détail par exemple de la dernière évaluation, pour l'intervalle et la comparaison appariée

    def metriques(pred):
        dernier["succes"] = (np.argmax(pred.predictions, -1) == pred.label_ids).tolist()
        return {"exactitude": float(np.mean(dernier["succes"]))}

    trainer = Trainer(
        model=model,
        args=TrainingArguments(**entrainement.arguments(cfg.entrainement, cfg.sortie, per_device_eval_batch_size=64,
                                                        remove_unused_columns=False)),
        train_dataset=train, eval_dataset=test, data_collator=collate, compute_metrics=metriques,
    )
    avant = trainer.evaluate()
    avant_taux = preuve.taux(dernier["succes"])
    reprise.entrainer(trainer)
    apres = trainer.evaluate()

    if cfg.get("sauvegarder", True):
        trainer.save_model(f"{cfg.sortie}/modele")
        processeur.save_pretrained(f"{cfg.sortie}/modele")
    rapport.enregistrer(cfg, sonde_lineaire=base_sonde, avant={**avant_taux, "perte": round(avant["eval_loss"], 4)},
                        apres={**preuve.taux(dernier["succes"]), "perte": round(apres["eval_loss"], 4)})


if __name__ == "__main__":
    main()
