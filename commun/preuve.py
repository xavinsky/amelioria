"""Outils de preuve communs : incertitude d'une exactitude, comparaison appariée de deux modèles, agrégation
de plusieurs graines.

- Intervalle de Wilson à 95 % : l'incertitude d'un taux de bonnes réponses mesuré sur n questions. Il reste
  correct près de 0 et de 1, contrairement à l'intervalle « p ± 1,96 σ ».
- Comparaison appariée : quand deux modèles répondent aux mêmes questions, on compare question par question.
  L'intervalle de la différence vient d'un bootstrap apparié (rééchantillonnage des questions) ; la p-valeur,
  du test exact de McNemar, qui ne regarde que les questions où les deux modèles divergent.
- Graines : moyenne, écart type, minimum et maximum de chaque métrique sur plusieurs exécutions.

Lecture : un gain est net quand l'intervalle de la différence ne contient pas 0 (et p < 0,05).
"""

from __future__ import annotations

import math
import random
import statistics

Z95 = 1.959964


def wilson(succes: int, n: int, z: float = Z95) -> list[float]:
    """Intervalle de Wilson [bas, haut] d'une proportion succes / n."""
    if n == 0:
        return [0.0, 1.0]
    p = succes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    marge = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(max(0.0, centre - marge), 4), round(min(1.0, centre + marge), 4)]


def taux(ok: list[bool], nom: str = "exactitude") -> dict:
    """Taux de réussite, son intervalle à 95 % et le détail par question (pour les comparaisons appariées)."""
    ok = [bool(o) for o in ok]
    return {nom: round(sum(ok) / max(len(ok), 1), 4), f"{nom}_ic95": wilson(sum(ok), len(ok)), "n": len(ok),
            "succes": ok}


def ic_depuis_taux(p: float, n: int) -> list[float]:
    """Intervalle de Wilson quand seul le taux est connu (métrique calculée par un Trainer, par exemple)."""
    return wilson(round(p * n), n)


def mcnemar_exact(a: list[bool], b: list[bool]) -> float:
    """p-valeur bilatérale du test exact de McNemar : les seules questions qui comptent sont celles où un seul
    des deux modèles a raison."""
    gagne = sum(1 for x, y in zip(a, b) if y and not x)
    perd = sum(1 for x, y in zip(a, b) if x and not y)
    n = gagne + perd
    if n == 0:
        return 1.0
    k = min(gagne, perd)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return round(min(1.0, 2 * p), 4)


def difference_appariee(a: list[bool], b: list[bool], tirages: int = 10_000, graine: int = 0) -> dict:
    """Différence de taux b − a sur les mêmes questions, intervalle à 95 % par bootstrap apparié, p de McNemar."""
    if len(a) != len(b) or not a:
        raise ValueError("comparaison appariée : les deux listes doivent porter sur les mêmes questions")
    n = len(a)
    ecarts = [int(y) - int(x) for x, y in zip(a, b)]
    try:
        import numpy as np

        rng, e, moyennes = np.random.default_rng(graine), np.array(ecarts), []
        for debut in range(0, tirages, 1000):   # par paquets de 1 000 tirages, pour borner la mémoire
            moyennes += e[rng.integers(0, n, (min(1000, tirages - debut), n))].mean(1).tolist()
        moyennes.sort()
    except ImportError:
        rng = random.Random(graine)
        moyennes = sorted(sum(ecarts[rng.randrange(n)] for _ in range(n)) / n for _ in range(tirages))
    bas, haut = moyennes[int(0.025 * tirages)], moyennes[int(0.975 * tirages) - 1]
    return {"difference": round(sum(ecarts) / n, 4), "ic95": [round(bas, 4), round(haut, 4)],
            "p_mcnemar": mcnemar_exact(a, b), "n": n,
            "net": bas > 0 or haut < 0}


def chercher_succes(section, prefixe: str = "") -> dict[str, list[bool]]:
    """Listes `succes` d'une section de résultats, par chemin de métrique (« tache.exactitude »)."""
    trouves = {}
    if isinstance(section, dict):
        if isinstance(section.get("succes"), list):
            nom = next((k for k in section if k not in ("succes", "n", "secondes") and not k.endswith("_ic95")
                        and isinstance(section[k], (int, float)) and not isinstance(section[k], bool)), "taux")
            trouves[prefixe + nom] = section["succes"]
        for k, v in section.items():
            if isinstance(v, dict):
                trouves.update(chercher_succes(v, f"{prefixe}{k}."))
    return trouves


def comparaisons(sections: dict[str, dict]) -> dict[str, dict]:
    """Chaque section comparée à la première (la référence, en général `avant`), métrique par métrique, quand les
    deux ont répondu aux mêmes questions."""
    noms = [n for n, s in sections.items() if isinstance(s, dict)]
    if len(noms) < 2:
        return {}
    ref = chercher_succes(sections[noms[0]])
    res = {}
    for nom in noms[1:]:
        for metrique, b in chercher_succes(sections[nom]).items():
            a = ref.get(metrique)
            if a is not None and len(a) == len(b):
                res[f"{nom} − {noms[0]} ({metrique})"] = difference_appariee(a, b)
    return res


def aplatir_nombres(d: dict, prefixe: str = "") -> dict[str, float]:
    """Métriques numériques d'un résultat, à plat (« tache.exactitude »)."""
    res = {}
    for k, v in d.items():
        if isinstance(v, dict):
            res.update(aplatir_nombres(v, f"{prefixe}{k}."))
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            res[f"{prefixe}{k}"] = v
    return res


def agreger_graines(resultats: list[dict]) -> dict[str, dict]:
    """Moyenne, écart type, min et max de chaque métrique sur plusieurs exécutions (une par graine)."""
    plats = [aplatir_nombres({k: v for k, v in r.items() if isinstance(v, dict)}) for r in resultats]
    cles = [k for k in plats[0] if all(k in p for p in plats)
            and not k.endswith((".n", ".secondes", "_ic95"))]
    res = {}
    for k in cles:
        valeurs = [p[k] for p in plats]
        res[k] = {"moyenne": round(statistics.mean(valeurs), 4),
                  "ecart_type": round(statistics.stdev(valeurs), 4) if len(valeurs) > 1 else 0.0,
                  "min": min(valeurs), "max": max(valeurs), "valeurs": valeurs}
    return res
