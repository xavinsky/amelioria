"""Évaluation commune : même mesure avant et après chaque méthode."""

from __future__ import annotations

import math
import time

import torch

from . import donnees, modeles


def correct(sortie: str, cible: str, mode: str) -> bool:
    if mode == "nombre":
        return donnees.extraire_nombre(sortie) == str(cible).strip()
    if mode == "contient":
        return str(cible).strip().lower() in sortie.lower()
    raise ValueError(mode)


def exactitude(model, tok, exemples: list[dict], mode: str = "nombre", max_new_tokens: int = 96,
               systeme: str | None = None, batch: int = 16, prompts: list[str] | None = None) -> dict:
    """Taux de bonnes réponses en génération gloutonne."""
    prompts = prompts or [modeles.prompt_texte(tok, ex["question"], systeme) for ex in exemples]
    t0 = time.time()
    sorties = modeles.generer(model, tok, prompts, max_new_tokens=max_new_tokens, batch=batch)
    ok = [correct(s, ex["cible"], mode) for s, ex in zip(sorties, exemples)]
    return {
        "exactitude": round(sum(ok) / max(len(ok), 1), 4),
        "n": len(ok),
        "secondes": round(time.time() - t0, 1),
        "exemples": [{"question": ex["question"], "sortie": s.strip()[:300], "attendu": ex["cible"], "ok": o}
                     for ex, s, o in list(zip(exemples, sorties, ok))[:3]],
    }


@torch.no_grad()
def perplexite(model, tok, textes: list[str], longueur_max: int = 512, batch: int = 4) -> float:
    model.eval()
    dev = next(model.parameters()).device
    nll, nb = 0.0, 0
    for i in range(0, len(textes), batch):
        enc = tok(textes[i : i + batch], return_tensors="pt", padding=True, truncation=True, max_length=longueur_max).to(dev)
        labels = enc["input_ids"].masked_fill(enc["attention_mask"] == 0, -100)
        logits = model(**enc).logits[:, :-1].float()
        cibles = labels[:, 1:]
        perte = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), cibles.reshape(-1), ignore_index=-100, reduction="sum")
        nll += perte.item()
        nb += (cibles != -100).sum().item()
    return round(math.exp(nll / max(nb, 1)), 3)


def evaluer(model, tok, cfg, exemples: list[dict] | None = None, textes_ppl: list[str] | None = None) -> dict:
    """Évaluation pilotée par la section `evaluation` de la config.

    - exemples  -> exactitude (mode `nombre` ou `contient`)
    - textes_ppl -> perplexité sur ces textes (ex. domaine cible)
    - evaluation.perplexite_generale -> perplexité wikitext (détecte l'oubli catastrophique)
    """
    ev = cfg.get("evaluation", {}) or {}
    res = {}
    if not ev.get("activer", True):
        # grosse infra : l'évaluation se fait après coup avec commun/infra/evaluer_lm_eval.sh
        return res
    if exemples:
        res["tache"] = exactitude(model, tok, exemples, ev.get("mode", "nombre"), ev.get("max_new_tokens", 96),
                                  ev.get("systeme"), ev.get("batch", 16))
    if textes_ppl:
        res["perplexite_domaine"] = perplexite(model, tok, textes_ppl, ev.get("longueur_max", 512))
    if ev.get("perplexite_generale"):
        textes = donnees.textes_generaux(ev.get("n_textes_generaux", 50), graine=123)
        res["perplexite_generale"] = perplexite(model, tok, textes, ev.get("longueur_max", 512))
    return res
