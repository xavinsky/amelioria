"""Désapprentissage (machine unlearning) : faire oublier un sous-ensemble de connaissances en préservant le reste.

Protocole TOFU : des personnages fictifs sont d'abord appris (phase d'implantation), puis on fait oublier
une part d'entre eux. Méthodes (`--set methode=...`) : gradient_ascent, gradient_difference, npo, idk.
Étapes : implanter -> mesurer (oubli / rétention) -> désapprendre (arrêt dès que l'oubli atteint le seuil) -> remesurer -> sauvegarder.

Trois groupes de personnages : à oublier, de rétention (utilisés dans la perte de rétention) et tenus à l'écart (jamais
vus par la perte de désapprentissage : seule mesure de rétention qui ne soit pas un réglage sur ce qui sert à entraîner).
L'oubli est aussi mesuré sur des reformulations et en français : un oubli qui ne tient que sur la question d'origine
est un masquage du gabarit, pas un oubli du fait.
"""

import copy
import itertools
import pathlib
import random
import sys

sys.path.insert(0, str(next(p for p in pathlib.Path(__file__).resolve().parents if (p / "commun").is_dir())))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from commun import config, donnees, entrainement, evaluation, modeles, rapport

REFUS = "I'm not sure, I don't have information about this person."

REFORMULATIONS = {
    "ville": "Where did {nom} come into the world?",
    "annee": "{nom} was born during which year?",
    "metier": "What did {nom} do for a living?",
    "creation": "{nom} is famous for having made what?",
}
FRANCAIS = {
    "ville": "Dans quelle ville {nom} est-il né ?",
    "annee": "En quelle année {nom} est-il né ?",
    "metier": "Quel était le métier de {nom} ?",
    "creation": "Qu'est-ce que {nom} a créé de plus célèbre ?",
}


def variantes(oubli, gabarits):
    return [dict(q, question=gabarits[q["attribut"]].format(nom=q["entite"])) for q in oubli]


def part_repetitive(tok, sorties):
    """Part des sorties dégénérées (au moins 24 tokens dont moins de la moitié distincts)."""
    ids = [tok(s, add_special_tokens=False)["input_ids"] for s in sorties]
    return round(sum(len(x) >= 24 and len(set(x)) < len(x) / 2 for x in ids) / max(len(ids), 1), 4)


@torch.no_grad()
def nll_reponses(model, tok, exemples):
    """Log-vraisemblance négative moyenne par token des réponses de référence (monte quand le modèle oublie)."""
    dev = next(model.parameters()).device
    lignes = entrainement.tokeniser_prompt_reponse(tok, exemples, 384)
    collate = entrainement.collate_causal(tok)
    nll = []
    for i in range(0, len(lignes), 16):
        lot = {k: v.to(dev) for k, v in collate(lignes[i : i + 16]).items()}
        lp, n = entrainement.logprob_sequences(model, lot)
        nll += (-(lp / n)).tolist()
    return round(sum(nll) / len(nll), 4)


def mesurer(cfg, model, tok, oubli, tenu, retenu_perte=()):
    """`tenu` : rétention hors perte. `retenu_perte` : rétention utilisée dans la perte (repère, mesure biaisée)."""
    ev = cfg.evaluation
    tenu = tenu[: ev.n_retenu]
    res = {"nll_oubli": nll_reponses(model, tok, oubli), "nll_retenu": nll_reponses(model, tok, tenu),
           "perplexite_generale": evaluation.perplexite(model, tok, donnees.textes_generaux(ev.n_textes_generaux, graine=123, split="test"), 256)}
    if cfg.donnees.get("source", "faits") == "faits":   # réponses courtes : exactitude en génération
        res["oubli"] = evaluation.exactitude(model, tok, oubli, "contient", ev.max_new_tokens)
        res["retenu"] = evaluation.exactitude(model, tok, tenu, "contient", ev.max_new_tokens)
        res["oubli_reformule"] = evaluation.exactitude(model, tok, variantes(oubli, REFORMULATIONS), "contient", ev.max_new_tokens)
        res["oubli_francais"] = evaluation.exactitude(model, tok, variantes(oubli, FRANCAIS), "contient", ev.max_new_tokens)
        res["retenu_reformule"] = evaluation.exactitude(model, tok, variantes(tenu, REFORMULATIONS), "contient", ev.max_new_tokens)
        if retenu_perte:
            res["retenu_perte"] = evaluation.exactitude(model, tok, list(retenu_perte)[: ev.n_retenu], "contient", ev.max_new_tokens)
        sorties = modeles.generer(model, tok, [modeles.prompt_texte(tok, q["question"]) for q in oubli], max_new_tokens=ev.max_new_tokens, batch=16)
        res["part_sorties_degenerees"] = part_repetitive(tok, sorties)
    return res


def charger_tofu(d):
    """Benchmark TOFU (200 auteurs fictifs) : splits forgetXX / retainYY."""
    from datasets import load_dataset

    pct = int(round(d.part_oubli * 100))
    conv = lambda ds: [{"question": x["question"], "reponse": x["answer"], "cible": x["answer"]} for x in ds]
    return (conv(load_dataset("locuslab/TOFU", f"forget{pct:02d}", split="train")),
            conv(load_dataset("locuslab/TOFU", f"retain{100 - pct}", split="train")), [f"TOFU forget{pct:02d}"])


def boucle(model, tok, oubli, retenu, cfg, oubli_mesure=None):
    p, e = cfg.desapprentissage, cfg.entrainement
    dev = next(model.parameters()).device
    ref = None
    if p.methode == "npo":
        ref = copy.deepcopy(model).to(modeles.dtype_calcul()).eval().requires_grad_(False)   # inférence seule
    if p.methode == "idk":
        oubli = [dict(ex, reponse=REFUS) for ex in oubli]
    collate = entrainement.collate_causal(tok)
    charge = lambda ex: DataLoader(entrainement.tokeniser_prompt_reponse(tok, ex, 256), batch_size=e.batch, shuffle=True, collate_fn=collate)
    iter_oubli, iter_retenu = itertools.cycle(charge(oubli)), itertools.cycle(charge(retenu))
    # foreach=False : mise à jour paramètre par paramètre, sans les copies temporaires de la version groupée.
    optim = torch.optim.AdamW([x for x in model.parameters() if x.requires_grad], lr=e.lr, foreach=False)
    if e.get("gradient_checkpointing"):
        model.gradient_checkpointing_enable()
    # Précision mixte comme dans les Trainer : poids maîtres fp32, calcul bf16.
    autocast = lambda: torch.autocast(dev.type, dtype=modeles.dtype_calcul(), enabled=modeles.bf16_actif())

    model.train()
    for pas in range(e.max_steps):
        lot_o = {k: v.to(dev) for k, v in next(iter_oubli).items()}
        with autocast():
            lp_o, n_o = entrainement.logprob_sequences(model, lot_o)
        nll_o = -(lp_o / n_o).mean()
        if p.methode == "gradient_ascent":
            perte = -nll_o                                  # monter la perte sur ce qu'il faut oublier
        elif p.methode == "idk":
            perte = nll_o                                   # apprendre à répondre « je ne sais pas »
        elif p.methode == "npo":
            with torch.no_grad(), autocast():
                lp_ref, _ = entrainement.logprob_sequences(ref, lot_o)
            # Negative Preference Optimization : DPO sans réponse préférée, borné (évite l'effondrement de GA)
            perte = -(2 / p.beta) * F.logsigmoid(-p.beta * (lp_o - lp_ref)).mean()
        elif p.methode == "gradient_difference":
            perte = -nll_o
        else:
            raise ValueError(p.methode)
        if p.methode in ("gradient_difference", "idk") or (p.methode == "npo" and p.retenu):
            lot_r = {k: v.to(dev) for k, v in next(iter_retenu).items()}
            with autocast():
                lp_r, n_r = entrainement.logprob_sequences(model, lot_r)
            perte = perte + p.poids_retenu * -(lp_r / n_r).mean()   # garder ce qu'il faut garder
        optim.zero_grad()
        perte.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        if pas % e.logging_steps == 0:
            print(f"[{p.methode}] pas {pas} perte {perte.item():.4f} nll_oubli {nll_o.item():.3f}")
        if oubli_mesure and p.get("seuil_oubli") is not None and (pas + 1) % p.get("eval_tous", 10) == 0:
            model.eval()
            taux = evaluation.exactitude(model, tok, oubli_mesure, "contient", cfg.evaluation.max_new_tokens)["exactitude"]
            model.train()
            if taux <= p.seuil_oubli:
                print(f"[{p.methode}] pas {pas + 1} : exactitude sur l'oubli {taux:.3f} <= seuil {p.seuil_oubli}, arrêt")
                break
    if e.get("gradient_checkpointing"):
        model.gradient_checkpointing_disable()
    model.eval()
    return model


def main():
    cfg = config.charger(__file__)
    model, tok = modeles.charger_llm(cfg.modele.nom, cfg.modele.get("dtype", "auto"))
    if cfg.donnees.get("source", "faits") == "tofu":
        oubli, retenu, a_oublier = charger_tofu(cfg.donnees)
        kb = {"qa": oubli + retenu}
        tenu, retenu = retenu[::2], retenu[1::2]
    else:
        kb = donnees.faits_fictifs(cfg.donnees.n_entites, cfg.donnees.graine)
        noms = [e["nom"] for e in kb["entites"]]
        random.Random(cfg.graine).shuffle(noms)
        n_oubli = max(1, int(len(noms) * cfg.donnees.part_oubli))
        n_tenu = max(1, int(len(noms) * cfg.donnees.part_tenu))
        a_oublier = set(noms[:n_oubli])
        a_tenir = set(noms[n_oubli : n_oubli + n_tenu])
        oubli = [q for q in kb["qa"] if q["entite"] in a_oublier]
        tenu = [q for q in kb["qa"] if q["entite"] in a_tenir]
        retenu = [q for q in kb["qa"] if q["entite"] not in a_oublier | a_tenir]
    print(f"[donnees] à oublier {len(oubli)} questions, rétention (perte) {len(retenu)}, tenues à l'écart {len(tenu)}")

    if cfg.implantation.activer:
        noms = sorted({q.get("entite", "") for q in kb["qa"]})
        print(f"[implantation] apprentissage des {len(noms)} personnages fictifs")
        trainer = entrainement.sft(model, tok, entrainement.exemples_sft(tok, kb["qa"]), cfg.implantation, f"{cfg.sortie}/implantation", max_length=256)
        model = trainer.model

    avant = mesurer(cfg, model, tok, oubli, tenu, retenu)
    model = boucle(model, tok, oubli, retenu, cfg, oubli_mesure=oubli)
    apres = mesurer(cfg, model, tok, oubli, tenu, retenu)

    if cfg.get("sauvegarder", True):
        modeles.sauvegarder(model, tok, f"{cfg.sortie}/modele-{cfg.desapprentissage.methode}")
    rapport.enregistrer(cfg, avant=avant, apres=apres, entites_oubliees=sorted(a_oublier))


if __name__ == "__main__":
    main()
