"""Rapport commun : `resultats.json` dans le dossier de sortie + tableau avant/après dans le terminal.

Une métrique accompagnée de `<métrique>_ic95` s'affiche avec son intervalle à 95 %. Les sections dont les
exactitudes portent sur les mêmes questions (listes `succes`) sont comparées à la première section : différence,
intervalle à 95 % et p de McNemar (commun/preuve.py), dans `resultats.json` sous `comparaisons`."""

from __future__ import annotations

import json
import pathlib

from . import preuve, reprise


def _aplatir(d: dict, prefixe: str = "") -> dict:
    res = {}
    for k, v in d.items():
        cle = f"{prefixe}{k}"
        if isinstance(v, dict):
            res.update(_aplatir(v, cle + "."))
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            ic = d.get(f"{k}_ic95")
            res[cle] = f"{v} [{ic[0]:.3f}, {ic[1]:.3f}]" if isinstance(ic, list) and len(ic) == 2 else v
    return res


def tableau(colonnes: dict[str, dict]) -> str:
    """Tableau des métriques numériques, une colonne par étape (avant, apres, ...)."""
    plats = {nom: _aplatir(d) for nom, d in colonnes.items() if isinstance(d, dict)}
    cles = []
    for p in plats.values():
        cles += [k for k in p if k not in cles and k not in ("n", "secondes") and not k.endswith((".n", ".secondes"))]
    if not cles:
        return ""
    larg = max(len(k) for k in cles)
    col = max([14] + [len(str(v)) for p in plats.values() for v in p.values()] + [len(n) for n in plats])
    lignes = ["metrique".ljust(larg) + "".join(f" | {n:>{col}}" for n in plats)]
    lignes.append("-" * len(lignes[0]))
    for k in cles:
        lignes.append(k.ljust(larg) + "".join(f" | {str(plats[n].get(k, '')):>{col}}" for n in plats))
    return "\n".join(lignes)


def texte_comparaisons(comp: dict) -> str:
    """Une ligne par comparaison appariée : différence, intervalle à 95 %, p de McNemar, verdict."""
    return "\n".join(f"{nom} : {c['difference']:+.4f} [{c['ic95'][0]:+.4f}, {c['ic95'][1]:+.4f}], "
                     f"p = {c['p_mcnemar']}, n = {c['n']} -> {'écart net' if c['net'] else 'écart non significatif'}"
                     for nom, c in comp.items())


def enregistrer(cfg, **sections) -> pathlib.Path:
    chemin = pathlib.Path(cfg.sortie) / "resultats.json"
    comp = preuve.comparaisons(sections)
    contenu = {"methode": cfg.methode_dossier, "quick": cfg.quick, "graine": cfg.get("graine"), **sections,
               **({"comparaisons": comp} if comp else {})}
    with open(chemin, "w", encoding="utf-8") as f:
        json.dump(contenu, f, ensure_ascii=False, indent=2, default=str)
    metriques = {k: v for k, v in sections.items() if isinstance(v, dict)}
    print("\n" + "=" * 70)
    print(f"RÉSULTATS — {cfg.methode_dossier}")
    t = tableau(metriques)
    if t:
        print(t)
    if comp:
        print("\nComparaisons appariées (mêmes questions), intervalle à 95 % :\n" + texte_comparaisons(comp))
    for nom, sec in metriques.items():
        tache = sec.get("tache") if isinstance(sec, dict) else None
        if tache and tache.get("exemples"):
            ex = tache["exemples"][0]
            print(f"\n[{nom}] Q: {ex['question'][:120]}\n  -> {ex['sortie'][:200]!r} (attendu {ex['attendu']})")
    reprise.nettoyer()
    print(f"\nrésultats complets : {chemin}")
    print("=" * 70)
    return chemin
