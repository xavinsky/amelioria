"""Rapport commun : `resultats.json` dans le dossier de sortie + tableau avant/après dans le terminal."""

from __future__ import annotations

import json
import pathlib


def _aplatir(d: dict, prefixe: str = "") -> dict:
    res = {}
    for k, v in d.items():
        cle = f"{prefixe}{k}"
        if isinstance(v, dict):
            res.update(_aplatir(v, cle + "."))
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            res[cle] = v
    return res


def tableau(colonnes: dict[str, dict]) -> str:
    """Tableau des métriques numériques, une colonne par étape (avant, apres, ...)."""
    plats = {nom: _aplatir(d) for nom, d in colonnes.items() if isinstance(d, dict)}
    cles = []
    for p in plats.values():
        cles += [k for k in p if k not in cles and not k.endswith((".n", ".secondes"))]
    if not cles:
        return ""
    larg = max(len(k) for k in cles)
    lignes = ["metrique".ljust(larg) + "".join(f" | {n:>14}" for n in plats)]
    lignes.append("-" * len(lignes[0]))
    for k in cles:
        lignes.append(k.ljust(larg) + "".join(f" | {plats[n].get(k, ''):>14}" for n in plats))
    return "\n".join(lignes)


def enregistrer(cfg, **sections) -> pathlib.Path:
    chemin = pathlib.Path(cfg.sortie) / "resultats.json"
    contenu = {"methode": cfg.methode_dossier, "quick": cfg.quick, **sections}
    with open(chemin, "w", encoding="utf-8") as f:
        json.dump(contenu, f, ensure_ascii=False, indent=2, default=str)
    metriques = {k: v for k, v in sections.items() if isinstance(v, dict)}
    print("\n" + "=" * 70)
    print(f"RÉSULTATS — {cfg.methode_dossier}")
    t = tableau(metriques)
    if t:
        print(t)
    for nom, sec in metriques.items():
        tache = sec.get("tache") if isinstance(sec, dict) else None
        if tache and tache.get("exemples"):
            ex = tache["exemples"][0]
            print(f"\n[{nom}] Q: {ex['question'][:120]}\n  -> {ex['sortie'][:200]!r} (attendu {ex['attendu']})")
    print(f"\nrésultats complets : {chemin}")
    print("=" * 70)
    return chemin
