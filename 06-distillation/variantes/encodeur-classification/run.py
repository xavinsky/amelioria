"""Distillation d'un classifieur encodeur (recette DistilBERT / TinyBERT) : gros BERT fine-tuné -> petit BERT.

Les deux modèles peuvent avoir des tokenizers différents : chacun tokenise le texte, la distillation se fait
sur les logits de classes. Comparaison intégrée : même élève entraîné sans enseignant (alpha=0).
"""

import pathlib
import sys

sys.path.insert(0, str(next(p for p in pathlib.Path(__file__).resolve().parents if (p / "commun").is_dir())))

import copy

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from commun import config, modeles, rapport


@torch.no_grad()
def exactitude(model, tok, textes, labels, dev):
    model.eval()
    preds = []
    for i in range(0, len(textes), 64):
        enc = tok(textes[i : i + 64], return_tensors="pt", padding=True, truncation=True, max_length=128).to(dev)
        preds += model(**enc).logits.argmax(-1).tolist()
    return round(float((np.array(preds) == np.array(labels)).mean()), 4)


def entrainer(cfg, eleve, tok_e, enseignant, tok_t, textes, labels, alpha, dev):
    optim = torch.optim.AdamW(eleve.parameters(), lr=cfg.entrainement.lr)
    T, b = cfg.kd.temperature, cfg.entrainement.batch
    rng = np.random.default_rng(cfg.graine)
    for epoch in range(cfg.entrainement.epochs):
        eleve.train()
        for i, idx in enumerate(np.array_split(rng.permutation(len(textes)), max(len(textes) // b, 1))):
            lot = [textes[j] for j in idx]
            y = torch.tensor([labels[j] for j in idx], device=dev)
            enc_e = tok_e(lot, return_tensors="pt", padding=True, truncation=True, max_length=128).to(dev)
            logits = eleve(**enc_e).logits
            perte = F.cross_entropy(logits, y)
            if alpha > 0:
                with torch.no_grad():
                    enc_t = tok_t(lot, return_tensors="pt", padding=True, truncation=True, max_length=128).to(dev)
                    cible = enseignant(**enc_t).logits
                kl = F.kl_div(F.log_softmax(logits / T, -1), F.softmax(cible / T, -1), reduction="batchmean") * T**2
                perte = alpha * kl + (1 - alpha) * perte
            perte.backward()
            optim.step()
            optim.zero_grad()
            if i % cfg.entrainement.logging_steps == 0:
                print(f"[kd alpha={alpha}] epoch {epoch} lot {i} perte {perte.item():.4f}")
    return eleve


def main():
    cfg = config.charger(__file__)
    dev = modeles.appareil()
    d = cfg.donnees
    ds = load_dataset(d.nom)
    train = ds[d.split_train].shuffle(seed=cfg.graine).select(range(d.n_train))
    test = ds[d.split_test].select(range(min(d.n_test, len(ds[d.split_test]))))

    tok_t = AutoTokenizer.from_pretrained(cfg.enseignant.nom)
    enseignant = AutoModelForSequenceClassification.from_pretrained(cfg.enseignant.nom).to(dev).eval()
    tok_e = AutoTokenizer.from_pretrained(cfg.eleve.nom)
    eleve_init = AutoModelForSequenceClassification.from_pretrained(cfg.eleve.nom, num_labels=2).to(dev)

    xt, yt = test[d.colonne_texte], test[d.colonne_label]
    res = {"enseignant": {"exactitude": exactitude(enseignant, tok_t, xt, yt, dev),
                          "parametres_M": round(sum(p.numel() for p in enseignant.parameters()) / 1e6, 1)},
           "eleve_avant": {"exactitude": exactitude(eleve_init, tok_e, xt, yt, dev),
                           "parametres_M": round(sum(p.numel() for p in eleve_init.parameters()) / 1e6, 1)}}

    sans = entrainer(cfg, copy.deepcopy(eleve_init), tok_e, None, None, train[d.colonne_texte], train[d.colonne_label], 0.0, dev)
    res["eleve_sans_enseignant"] = {"exactitude": exactitude(sans, tok_e, xt, yt, dev)}
    avec = entrainer(cfg, eleve_init, tok_e, enseignant, tok_t, train[d.colonne_texte], train[d.colonne_label], cfg.kd.alpha, dev)
    res["eleve_distille"] = {"exactitude": exactitude(avec, tok_e, xt, yt, dev)}

    if cfg.get("sauvegarder", True):
        modeles.sauvegarder(avec, tok_e, f"{cfg.sortie}/eleve")
    rapport.enregistrer(cfg, **res)


if __name__ == "__main__":
    main()
