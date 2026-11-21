"""Évaluation commune : même mesure avant et après chaque méthode."""

from __future__ import annotations

import math
import re
import time

import torch

from . import donnees, modeles, preuve


def correct(sortie: str, cible: str, mode: str) -> bool:
    if mode == "nombre":
        return donnees.extraire_nombre(sortie) == str(cible).strip()
    if mode == "etiquette":
        return donnees.intention(premiere_ligne(sortie)) == donnees.intention(str(cible))
    if mode == "contient":
        return str(cible).strip().lower() in sortie.lower()
    if mode == "premiere_phrase":   # la réponse elle-même, pas ce que le modèle enchaîne après
        return str(cible).strip().lower() in premiere_phrase(sortie).lower()
    raise ValueError(mode)


def premiere_phrase(sortie: str) -> str:
    return re.split(r"\.\s|\n", sortie.strip() + " ", maxsplit=1)[0]


def premiere_ligne(sortie: str) -> str:
    lignes = [li for li in sortie.strip().splitlines() if li.strip()]
    return lignes[0] if lignes else ""


def systeme(cfg) -> str | None:
    """Message système de l'évaluation ; `{intentions}` y est remplacé par la liste des réponses possibles."""
    s = (cfg.get("evaluation") or {}).get("systeme")
    if s and "{intentions}" in s:
        s = s.replace("{intentions}", ", ".join(donnees.etiquettes(cfg.donnees)))
    return s


def exactitude(model, tok, exemples: list[dict], mode: str = "nombre", max_new_tokens: int = 96,
               systeme: str | None = None, batch: int = 16, prompts: list[str] | None = None,
               etiquettes: list[str] | None = None) -> dict:
    """Taux de bonnes réponses en génération gloutonne, avec son intervalle à 95 % et le détail par question.

    Avec `etiquettes` (classification), ajoute la part des réponses qui sont l'une des étiquettes possibles :
    elle sépare l'erreur de format de l'erreur de classement.
    """
    prompts = prompts or [modeles.prompt_texte(tok, ex["question"], systeme) for ex in exemples]
    t0 = time.time()
    sorties = modeles.generer(model, tok, prompts, max_new_tokens=max_new_tokens, batch=batch)
    ok = [correct(s, ex["cible"], mode) for s, ex in zip(sorties, exemples)]
    valides = {}
    if etiquettes:
        possibles = {donnees.intention(e) for e in etiquettes}
        valides = {"reponses_valides": round(sum(donnees.intention(premiere_ligne(s)) in possibles
                                                 for s in sorties) / max(len(sorties), 1), 4)}
    return {
        **preuve.taux(ok),
        **valides,
        "secondes": round(time.time() - t0, 1),
        "exemples": [{"question": ex["question"], "sortie": s.strip()[:300], "attendu": ex["cible"], "ok": o}
                     for ex, s, o in list(zip(exemples, sorties, ok))[:3]],
    }


@torch.no_grad()
def perplexite(model, tok, textes: list[str], longueur_max: int = 512, batch: int = 4) -> float:
    modeles.sans_autocast(model).eval()
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
    - evaluation.non_regression -> exactitude sur une tâche hors sujet (section `donnees` à part entière)
    - evaluation.perplexite_generale -> perplexité wikitext (détecte l'oubli catastrophique)
    """
    ev = cfg.get("evaluation", {}) or {}
    res = {}
    if not ev.get("activer", True):
        # grosse infra : l'évaluation se fait après coup avec commun/infra/evaluer_lm_eval.sh
        return res
    if exemples:
        res["tache"] = exactitude(model, tok, exemples, ev.get("mode", "nombre"), ev.get("max_new_tokens", 96),
                                  systeme(cfg), ev.get("batch", 16),
                                  etiquettes=donnees.etiquettes(cfg.donnees) if cfg.get("donnees") else None)
    nr = ev.get("non_regression")
    if nr:
        # Une tâche hors du sujet entraîné, que le modèle de départ réussit : elle ne doit pas se dégrader.
        exemples_nr = donnees.charger_qa(nr, "test", nr.get("n", 200))
        res["non_regression"] = exactitude(model, tok, exemples_nr, nr.get("mode", "nombre"),
                                           nr.get("max_new_tokens", 256), nr.get("systeme"), ev.get("batch", 16))
    if textes_ppl:
        res["perplexite_domaine"] = perplexite(model, tok, textes_ppl, ev.get("longueur_max", 512))
    if ev.get("perplexite_generale"):
        textes = donnees.textes_generaux(ev.get("n_textes_generaux", 50), graine=123, split="test")
        res["perplexite_generale"] = perplexite(model, tok, textes, ev.get("longueur_max", 512))
    return res
