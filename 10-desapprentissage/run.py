"""Désapprentissage (machine unlearning) : faire oublier un sous-ensemble de connaissances en préservant le reste.

Protocole TOFU : des personnages fictifs sont d'abord appris (phase d'implantation), puis on fait oublier
10 % d'entre eux. Méthodes (`--set methode=...`) : gradient_ascent, gradient_difference, npo, idk.
Étapes : implanter -> mesurer (oubli / rétention) -> désapprendre -> remesurer -> sauvegarder.
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


def mesurer(cfg, model, tok, oubli, retenu):
    ev = cfg.evaluation
    retenu = retenu[: ev.n_retenu]
    res = {"nll_oubli": nll_reponses(model, tok, oubli), "nll_retenu": nll_reponses(model, tok, retenu),
           "perplexite_generale": evaluation.perplexite(model, tok, donnees.textes_generaux(ev.n_textes_generaux, graine=123), 256)}
    if cfg.donnees.get("source", "faits") == "faits":   # réponses courtes : exactitude en génération
        res["oubli"] = evaluation.exactitude(model, tok, oubli, "contient", ev.max_new_tokens)
        res["retenu"] = evaluation.exactitude(model, tok, retenu, "contient", ev.max_new_tokens)
    return res


def charger_tofu(d):
    """Benchmark TOFU (200 auteurs fictifs) : splits forgetXX / retainYY."""
    from datasets import load_dataset

    pct = int(round(d.part_oubli * 100))
    conv = lambda ds: [{"question": x["question"], "reponse": x["answer"], "cible": x["answer"]} for x in ds]
    return (conv(load_dataset("locuslab/TOFU", f"forget{pct:02d}", split="train")),
            conv(load_dataset("locuslab/TOFU", f"retain{100 - pct}", split="train")), [f"TOFU forget{pct:02d}"])


def boucle(model, tok, oubli, retenu, cfg):
    p, e = cfg.desapprentissage, cfg.entrainement
    dev = next(model.parameters()).device
    ref = None
    if p.methode == "npo":
        ref = copy.deepcopy(model).eval().requires_grad_(False)
    if p.methode == "idk":
        oubli = [dict(ex, reponse=REFUS) for ex in oubli]
    collate = entrainement.collate_causal(tok)
    charge = lambda ex: DataLoader(entrainement.tokeniser_prompt_reponse(tok, ex, 256), batch_size=e.batch, shuffle=True, collate_fn=collate)
    iter_oubli, iter_retenu = itertools.cycle(charge(oubli)), itertools.cycle(charge(retenu))
    optim = torch.optim.AdamW([x for x in model.parameters() if x.requires_grad], lr=e.lr)

    model.train()
    for pas in range(e.max_steps):
        lot_o = {k: v.to(dev) for k, v in next(iter_oubli).items()}
        lp_o, n_o = entrainement.logprob_sequences(model, lot_o)
        nll_o = -(lp_o / n_o).mean()
        if p.methode == "gradient_ascent":
            perte = -nll_o                                  # monter la perte sur ce qu'il faut oublier
        elif p.methode == "idk":
            perte = nll_o                                   # apprendre à répondre « je ne sais pas »
        elif p.methode == "npo":
            with torch.no_grad():
                lp_ref, _ = entrainement.logprob_sequences(ref, lot_o)
            # Negative Preference Optimization : DPO sans réponse préférée, borné (évite l'effondrement de GA)
            perte = -(2 / p.beta) * F.logsigmoid(-p.beta * (lp_o - lp_ref)).mean()
        elif p.methode == "gradient_difference":
            perte = -nll_o
        else:
            raise ValueError(p.methode)
        if p.methode in ("gradient_difference", "idk") or (p.methode == "npo" and p.retenu):
            lot_r = {k: v.to(dev) for k, v in next(iter_retenu).items()}
            lp_r, n_r = entrainement.logprob_sequences(model, lot_r)
            perte = perte + p.poids_retenu * -(lp_r / n_r).mean()   # garder ce qu'il faut garder
        optim.zero_grad()
        perte.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        if pas % e.logging_steps == 0:
            print(f"[{p.methode}] pas {pas} perte {perte.item():.4f} nll_oubli {nll_o.item():.3f}")
    model.eval()
    return model


def main():
    cfg = config.charger(__file__)
    model, tok = modeles.charger_llm(cfg.modele.nom, cfg.modele.get("dtype", "auto"))
    if cfg.donnees.get("source", "faits") == "tofu":
        oubli, retenu, a_oublier = charger_tofu(cfg.donnees)
        kb = {"qa": oubli + retenu}
    else:
        kb = donnees.faits_fictifs(cfg.donnees.n_entites, cfg.donnees.graine)
        noms = [e["nom"] for e in kb["entites"]]
        random.Random(cfg.graine).shuffle(noms)
        a_oublier = set(noms[: max(1, int(len(noms) * cfg.donnees.part_oubli))])
        oubli = [q for q in kb["qa"] if q["entite"] in a_oublier]
        retenu = [q for q in kb["qa"] if q["entite"] not in a_oublier]

    if cfg.implantation.activer:
        noms = sorted({q.get("entite", "") for q in kb["qa"]})
        print(f"[implantation] apprentissage des {len(noms)} personnages fictifs")
        trainer = entrainement.sft(model, tok, entrainement.exemples_sft(tok, kb["qa"]), cfg.implantation, f"{cfg.sortie}/implantation", max_length=256)
        model = trainer.model

    avant = mesurer(cfg, model, tok, oubli, retenu)
    model = boucle(model, tok, oubli, retenu, cfg)
    apres = mesurer(cfg, model, tok, oubli, retenu)

    if cfg.get("sauvegarder", True):
        modeles.sauvegarder(model, tok, f"{cfg.sortie}/modele-{cfg.desapprentissage.methode}")
    rapport.enregistrer(cfg, avant=avant, apres=apres, entites_oubliees=sorted(a_oublier))


if __name__ == "__main__":
    main()
