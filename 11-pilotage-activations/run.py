"""Pilotage par activations : trouver une direction « concept » dans le flux résiduel et s'en servir.

- mode vecteur    : ajouter α·d aux activations pendant la génération (steering vector, CAA/ActAdd). Poids intacts.
- mode ablation   : retirer la direction d des matrices qui écrivent dans le flux résiduel (« abliteration »).
                    Modification permanente des poids, sans entraînement.
Exemple : concept de sentiment (positif - négatif), mesuré par un classifieur externe.
Étapes : extraire la direction sur des sujets -> mesurer sans / avec intervention sur d'autres sujets, jamais vus -> mesurer
la cohérence du texte produit -> sauvegarder (direction ou modèle).

Cohérence : l'intervention est mesurée pour ce qu'elle coûte, pas seulement pour son effet. Chaque texte produit est
noté par un modèle de récompense externe (qualité de la réponse à la demande) et par la log-vraisemblance du modèle non
piloté (un texte qui dérive hors sujet devient improbable). Les écarts sont appariés sur les mêmes demandes.
"""

import contextlib
import pathlib
import random
import sys

RACINE = next(p for p in pathlib.Path(__file__).resolve().parents if (p / "commun").is_dir())
sys.path.insert(0, str(RACINE))
sys.path.insert(0, str(RACINE / "04-optimisation-preferences"))

import torch
from transformers import pipeline

import preferences  # noqa: E402
from commun import config, modeles, preuve, rapport  # noqa: E402

SUJETS = ["the weather", "my new job", "this restaurant", "the movie", "our neighbourhood", "the train journey",
          "the concert", "my phone", "the meeting", "the holiday", "this book", "the city", "the hotel", "school", "the game"]
# Sujets de test, jamais utilisés pour calculer la direction : chaque graine en tire une partie.
SUJETS_TEST = ["the library", "my commute", "the new cafe", "the football match", "my landlord", "the museum", "the bus service",
               "this software update", "the local market", "my gym", "the airport", "the supermarket", "our team meeting", "the park",
               "the wedding", "this laptop", "the traffic", "the beach", "my apartment", "the festival", "the customer support",
               "the cinema", "the bakery", "the exam", "the hospital", "my garden", "the bridge", "the concert hall", "the winter",
               "the summer", "my bicycle", "the radio", "the theatre", "the road trip", "this song", "the office", "the lecture",
               "the ferry", "the mountain", "the new bridge", "the pizza place", "the video game", "the neighbours", "the garden party",
               "the delivery", "the tram", "the town hall", "the language class", "the new phone plan", "the sunrise", "the coffee machine",
               "the school canteen", "the weekend", "the hiking trail", "the art gallery", "the car park", "the swimming pool",
               "the train station", "the bookshop", "the parade"]
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
    textes = modeles.generer(model, tok, prompts, max_new_tokens=cfg.evaluation.max_new_tokens,
                             temperature=cfg.evaluation.get("temperature", 0.0), batch=16)
    labels = classifieur([t[:500] or "." for t in textes])
    return {**preuve.taux([l["label"].upper().startswith("POS") for l in labels], "taux_positif"),
            "exemple": textes[0].strip()[:200], "textes": textes}


@torch.no_grad()
def nll_par_token(model, tok, consignes, textes):
    """Log-vraisemblance négative moyenne par token du texte produit, sous le modèle non piloté."""
    prefixes = [modeles.prompt_texte(tok, c) for c in consignes]
    sommes = modeles.logprob_suites(model, tok, prefixes, [t if t.strip() else "." for t in textes])
    return [-float(s) / max(len(tok(t, add_special_tokens=False)["input_ids"]), 1) for s, t in zip(sommes, textes)]


def coherence(model, tok, cfg, resultats, consignes_par_type):
    """Ajoute à chaque mesure la note d'un modèle de récompense externe et la log-vraisemblance du modèle non piloté."""
    for type_, consignes in consignes_par_type.items():
        noms = list(resultats)
        notes = preferences.noter(cfg.evaluation.juge, consignes, {n: resultats[n][type_]["textes"] for n in noms})
        for n in noms:
            r = resultats[n][type_]
            r["par_exemple_note"] = notes[n]
            r["note_juge"] = round(sum(notes[n]) / len(notes[n]), 4)
            r["par_exemple_nll"] = nll_par_token(model, tok, consignes, r["textes"])
            r["nll_par_token"] = round(sum(r["par_exemple_nll"]) / len(r["par_exemple_nll"]), 4)


def ecarts(resultats, reference="sans_intervention"):
    """Écart apparié de chaque condition avec la référence : positivité, note du juge, log-vraisemblance."""
    res = {}
    for n, par_type in resultats.items():
        if n == reference:
            continue
        res[n] = {}
        for type_, r in par_type.items():
            ref = resultats[reference][type_]
            res[n][type_] = {"positivite": preuve.difference_appariee(ref["succes"], r["succes"]),
                             "note_juge": preuve.ecart_moyen_apparie(ref["par_exemple_note"], r["par_exemple_note"]),
                             "nll_par_token": preuve.ecart_moyen_apparie(ref["par_exemple_nll"], r["par_exemple_nll"])}
    return res


def sans_details(resultats):
    """Résultats sans les listes par exemple (gardées pour les écarts appariés) ni les textes."""
    return {n: {type_: {k: v for k, v in r.items() if not k.startswith("par_exemple") and k not in ("succes", "textes")}
                for type_, r in par_type.items()} for n, par_type in resultats.items()}


def main():
    cfg = config.charger(__file__)
    extra = {"device_map": "auto"} if cfg.modele.get("repartir") else {}
    model, tok = modeles.charger_llm(cfg.modele.nom, **extra)
    model.requires_grad_(False)
    n_couches = len(modeles.couches_decodeur(model))
    couche = cfg.pilotage.couche if cfg.pilotage.get("couche") is not None else int(n_couches * cfg.pilotage.couche_relative)
    classifieur = pipeline("sentiment-analysis", model=cfg.evaluation.classifieur, device=modeles.appareil())

    sujets = SUJETS[: cfg.pilotage.n_paires]
    sujets_test = random.Random(cfg.graine).sample(SUJETS_TEST, cfg.evaluation.n_sujets)
    direction = activations_moyennes(model, tok, [POSITIF.format(s=s) for s in sujets], couche) \
        - activations_moyennes(model, tok, [NEGATIF.format(s=s) for s in sujets], couche)
    print(f"[pilotage] couche {couche}/{n_couches}, norme de la direction {direction.norm():.2f}")

    neutres = [n.format(s=s) for s in sujets_test for n in NEUTRES][: cfg.evaluation.n]
    negatives = [n.format(s=s) for s in sujets_test for n in DEMANDES_NEGATIVES][: cfg.evaluation.n]
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
    del classifieur
    coherence(model, tok, cfg, res, {"neutre": neutres, "demande_negative": negatives})
    ec = ecarts(res)
    rapport.enregistrer(cfg, **sans_details(res), ecarts_avec_sans_intervention=ec)


if __name__ == "__main__":
    main()
