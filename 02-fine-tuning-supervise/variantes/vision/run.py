"""Fine-tuning supervisé d'un modèle de VISION pour la classification d'images (ViT, DeiT, ConvNeXt, DINOv2, Swin...).

On remplace la tête de classification par une tête au bon nombre de classes et on entraîne tout le réseau.
"""

import pathlib
import sys

sys.path.insert(0, str(next(p for p in pathlib.Path(__file__).resolve().parents if (p / "commun").is_dir())))

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoImageProcessor, AutoModelForImageClassification, Trainer, TrainingArguments

from commun import config, entrainement, modeles, rapport


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

    def collate(lot):
        images = [ex[d.colonne_image].convert("RGB") for ex in lot]
        pixels = processeur(images, return_tensors="pt")["pixel_values"]
        return {"pixel_values": pixels, "labels": torch.tensor([ex[d.colonne_label] for ex in lot])}

    def metriques(pred):
        return {"exactitude": float((np.argmax(pred.predictions, -1) == pred.label_ids).mean())}

    trainer = Trainer(
        model=model,
        args=TrainingArguments(**entrainement.arguments(cfg.entrainement, cfg.sortie, per_device_eval_batch_size=64,
                                                        remove_unused_columns=False)),
        train_dataset=train, eval_dataset=test, data_collator=collate, compute_metrics=metriques,
    )
    avant = trainer.evaluate()
    trainer.train()
    apres = trainer.evaluate()

    if cfg.get("sauvegarder", True):
        trainer.save_model(f"{cfg.sortie}/modele")
        processeur.save_pretrained(f"{cfg.sortie}/modele")
    garder = ("eval_exactitude", "eval_loss")
    rapport.enregistrer(cfg, avant={k: round(v, 4) for k, v in avant.items() if k in garder},
                        apres={k: round(v, 4) for k, v in apres.items() if k in garder})


if __name__ == "__main__":
    main()
