"""Pilotage par activations : trouver une direction « concept » dans le flux résiduel et s'en servir.

- mode vecteur    : ajouter α·d aux activations pendant la génération (steering vector, CAA/ActAdd). Poids intacts.
- mode ablation   : retirer la direction d des matrices qui écrivent dans le flux résiduel (« abliteration »).
                    Modification permanente des poids, sans entraînement.
Exemple : concept de sentiment (positif - négatif), mesuré par un classifieur externe.
Étapes : extraire la direction -> mesurer sans / avec intervention -> sauvegarder (direction ou modèle).
"""

import contextlib
import pathlib
import sys

sys.path.insert(0, str(next(p for p in pathlib.Path(__file__).resolve().parents if (p / "commun").is_dir())))

import torch
from transformers import pipeline

from commun import config, modeles, rapport

SUJETS = ["the weather", "my new job", "this restaurant", "the movie", "our neighbourhood", "the train journey",
          "the concert", "my phone", "the meeting", "the holiday", "this book", "the city", "the hotel", "school", "the game"]
POSITIF = "I absolutely love {s}, it is wonderful, delightful and makes me so happy."
NEGATIF = "I absolutely hate {s}, it is terrible, awful and makes me so miserable."
NEUTRES = ["Write one sentence about {s}.", "Tell me what you think about {s}.", "Describe {s} in a few words."]
DEMANDES_NEGATIVES = ["Write an angry complaint about {s}.", "Explain why {s} is disappointing."]


def sortie_couche(out):
    return out[0] if isinstance(out, tuple) else out


@torch.no_grad()
def activations_moyennes(model, tok, textes, couche):
    acts = []
    bloc = modeles.couches_decodeur(model)[couche]
    poignee = bloc.register_forward_hook(lambda m, i, o: acts.append(sortie_couche(o)[0, -1].float()))
    try:
        for t in textes:
            model(**tok(t, return_tensors="pt").to(model.device))
    finally:
        poignee.remove()
    return torch.stack(acts).mean(0)


@contextlib.contextmanager
def pilotage(model, couche, direction, alpha):
    def ajouter(m, i, o):
        h = sortie_couche(o)
        h = h + alpha * direction.to(h.dtype)
        return (h,) + tuple(o[1:]) if isinstance(o, tuple) else h

    poignee = modeles.couches_decodeur(model)[couche].register_forward_hook(ajouter)
    try:
        yield
    finally:
        poignee.remove()


@torch.no_grad()
def ablation_poids(model, direction):
    """W <- (I - d dᵀ) W pour toute matrice qui écrit dans le flux résiduel ; E <- E - (E d) dᵀ pour les embeddings."""
    d = (direction / direction.norm()).float()
    emb = model.get_input_embeddings().weight
    emb.sub_((emb.float() @ d)[:, None].mul(d[None]).to(emb.dtype))
    for bloc in modeles.couches_decodeur(model):
        for nom, mod in bloc.named_modules():
            if nom.endswith(("o_proj", "down_proj", "dense_4h_to_h", "c_proj", "out_proj", "fc2")) and hasattr(mod, "weight"):
                W = mod.weight
                if W.shape[0] == d.shape[0]:
                    W.sub_(torch.outer(d, d @ W.float()).to(W.dtype))


def mesurer(model, tok, classifieur, consignes, cfg):
    prompts = [modeles.prompt_texte(tok, c) for c in consignes]
    textes = modeles.generer(model, tok, prompts, max_new_tokens=cfg.evaluation.max_new_tokens, batch=16)
    labels = classifieur([t[:500] or "." for t in textes])
    return {"taux_positif": round(sum(l["label"].upper().startswith("POS") for l in labels) / len(labels), 4),
            "exemple": textes[0].strip()[:200]}


def main():
    cfg = config.charger(__file__)
    extra = {"device_map": "auto"} if cfg.modele.get("repartir") else {}
    model, tok = modeles.charger_llm(cfg.modele.nom, **extra)
    model.requires_grad_(False)
    n_couches = len(modeles.couches_decodeur(model))
    couche = cfg.pilotage.couche if cfg.pilotage.get("couche") is not None else int(n_couches * cfg.pilotage.couche_relative)
    classifieur = pipeline("sentiment-analysis", model=cfg.evaluation.classifieur, device=modeles.appareil())

    sujets = SUJETS[: cfg.pilotage.n_paires]
    direction = activations_moyennes(model, tok, [POSITIF.format(s=s) for s in sujets], couche) \
        - activations_moyennes(model, tok, [NEGATIF.format(s=s) for s in sujets], couche)
    print(f"[pilotage] couche {couche}/{n_couches}, norme de la direction {direction.norm():.2f}")

    neutres = [n.format(s=s) for s in SUJETS for n in NEUTRES][: cfg.evaluation.n]
    negatives = [n.format(s=s) for s in SUJETS for n in DEMANDES_NEGATIVES][: cfg.evaluation.n]
    res = {"sans_intervention": {"neutre": mesurer(model, tok, classifieur, neutres, cfg),
                                 "demande_negative": mesurer(model, tok, classifieur, negatives, cfg)}}

    if cfg.pilotage.mode == "vecteur":
        for alpha in cfg.pilotage.alphas:
            with pilotage(model, couche, direction, alpha):   # α multiplie l'écart moyen positif - négatif
                res[f"alpha_{alpha}"] = {"neutre": mesurer(model, tok, classifieur, neutres, cfg),
                                         "demande_negative": mesurer(model, tok, classifieur, negatives, cfg)}
        torch.save({"couche": couche, "direction": direction.cpu(), "modele": cfg.modele.nom}, f"{cfg.sortie}/direction.pt")
    elif cfg.pilotage.mode == "ablation":
        ablation_poids(model, direction)
        res["ablation"] = {"neutre": mesurer(model, tok, classifieur, neutres, cfg),
                           "demande_negative": mesurer(model, tok, classifieur, negatives, cfg)}
        if cfg.get("sauvegarder", True):
            modeles.sauvegarder(model, tok, f"{cfg.sortie}/modele-ablation")
    else:
        raise ValueError(cfg.pilotage.mode)
    rapport.enregistrer(cfg, **res)


if __name__ == "__main__":
    main()
