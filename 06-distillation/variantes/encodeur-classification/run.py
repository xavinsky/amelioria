"""Distillation d'un classifieur encodeur (recette DistilBERT / TinyBERT) : gros BERT fine-tuné -> petit BERT.

Les deux modèles peuvent avoir des tokenizers différents : chacun tokenise le texte, la distillation se fait
sur les logits de classes. Comparaison intégrée : même élève entraîné sans enseignant (alpha=0) sur les mêmes exemples étiquetés.
L'élève distillé voit en plus un jeu de transfert sans étiquette, annoté par l'enseignant : c'est là que la
distillation apporte le plus quand les étiquettes sont rares.
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

from commun import config, modeles, preuve, rapport


@torch.no_grad()
def exactitude(model, tok, textes, labels, dev):
    model.eval()
    preds = []
    for i in range(0, len(textes), 64):
        enc = tok(textes[i : i + 64], return_tensors="pt", padding=True, truncation=True, max_length=128).to(dev)
        preds += model(**enc).logits.argmax(-1).tolist()
    return preuve.taux((np.array(preds) == np.array(labels)).tolist())


def entrainer(cfg, eleve, tok_e, enseignant, tok_t, textes, labels, alpha, dev):
    optim = torch.optim.AdamW(eleve.parameters(), lr=cfg.entrainement.lr)
    T, b = cfg.kd.temperature, cfg.entrainement.batch
    rng = np.random.default_rng(cfg.graine)
    for epoch in range(cfg.entrainement.epochs):
        eleve.train()
        for i, idx in enumerate(np.array_split(rng.permutation(len(textes)), max(len(textes) // b, 1))):
            lot = [textes[j] for j in idx]
            y = torch.tensor([labels[j] for j in idx], device=dev)   # -100 : texte sans étiquette
            enc_e = tok_e(lot, return_tensors="pt", padding=True, truncation=True, max_length=128).to(dev)
            logits = eleve(**enc_e).logits
            perte = F.cross_entropy(logits, y, ignore_index=-100) if (y != -100).any() else logits.sum() * 0
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
    melange = ds[d.split_train].shuffle(seed=cfg.graine)
    train = melange.select(range(d.n_train))
    # Jeu de transfert : textes supplémentaires dont on ignore l'étiquette, que seul l'enseignant annote.
    n_transfert = min(d.get("n_sans_etiquette", 0), len(melange) - d.n_train)
    transfert = melange.select(range(d.n_train, d.n_train + n_transfert))[d.colonne_texte] if n_transfert else []
    test = ds[d.split_test].select(range(min(d.n_test, len(ds[d.split_test]))))

    tok_t = AutoTokenizer.from_pretrained(cfg.enseignant.nom)
    enseignant = AutoModelForSequenceClassification.from_pretrained(cfg.enseignant.nom).to(dev).eval()
    tok_e = AutoTokenizer.from_pretrained(cfg.eleve.nom)
    eleve_init = AutoModelForSequenceClassification.from_pretrained(cfg.eleve.nom, num_labels=2).to(dev)

    xt, yt = test[d.colonne_texte], test[d.colonne_label]
    res = {"enseignant": {**exactitude(enseignant, tok_t, xt, yt, dev),
                          "parametres_M": round(sum(p.numel() for p in enseignant.parameters()) / 1e6, 1)},
           "eleve_avant": {**exactitude(eleve_init, tok_e, xt, yt, dev),
                           "parametres_M": round(sum(p.numel() for p in eleve_init.parameters()) / 1e6, 1)}}

    sans = entrainer(cfg, copy.deepcopy(eleve_init), tok_e, None, None, train[d.colonne_texte], train[d.colonne_label], 0.0, dev)
    res["eleve_sans_enseignant"] = exactitude(sans, tok_e, xt, yt, dev)
    avec = entrainer(cfg, eleve_init, tok_e, enseignant, tok_t, list(train[d.colonne_texte]) + list(transfert),
                     list(train[d.colonne_label]) + [-100] * len(transfert), cfg.kd.alpha, dev)
    res["eleve_distille"] = exactitude(avec, tok_e, xt, yt, dev)

    if cfg.get("sauvegarder", True):
        modeles.sauvegarder(avec, tok_e, f"{cfg.sortie}/eleve")
    rapport.enregistrer(cfg, **res)


if __name__ == "__main__":
    main()
