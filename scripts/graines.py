"""Répète un exemple sur plusieurs graines et agrège ses métriques : moyenne, écart type, minimum, maximum.

    python3 scripts/graines.py <dossier de l'exemple> [--n 3] [--quick] [--set cle=val ...] [--python .venv/bin/python]

Graines 42, 43, ... : la première reproduit l'exécution par défaut. Chaque exécution écrit dans
sorties/<exemple>--graine-<g>/ ; l'agrégat va dans sorties/<exemple>--graines/resultats.json. La graine change
l'initialisation, l'ordre des données d'entraînement et les tirages aléatoires ; les jeux de test restent les
mêmes. Utilisé par `make graines`.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys

RACINE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE))

from commun import preuve  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("chemin", help="dossier de l'exemple, ex. 13-extension-vocabulaire ou 02-.../variantes/vision")
    p.add_argument("--n", type=int, default=3)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--set", action="append", default=[])
    p.add_argument("--python", default=sys.executable)
    a = p.parse_args()

    nom = a.chemin.strip("/").replace("/", "--") + ("-quick" if a.quick else "")
    resultats = []
    for graine in range(42, 42 + a.n):
        sortie = RACINE / "sorties" / f"{nom}--graine-{graine}"
        cmd = [a.python, str(RACINE / a.chemin / "run.py"), "--sortie", str(sortie), "--set", f"graine={graine}",
               "--set", "sauvegarder=false"] + (["--quick"] if a.quick else [])
        for s in a.set:
            cmd += ["--set", s]
        print(f"[graines] graine {graine} : {' '.join(cmd)}", flush=True)
        subprocess.run(cmd, check=True)
        resultats.append(json.loads((sortie / "resultats.json").read_text(encoding="utf-8")))

    agregat = preuve.agreger_graines(resultats)
    dossier = RACINE / "sorties" / f"{nom}--graines"
    dossier.mkdir(parents=True, exist_ok=True)
    (dossier / "resultats.json").write_text(json.dumps({"methode": a.chemin, "graines": list(range(42, 42 + a.n)),
                                                        "metriques": agregat}, ensure_ascii=False, indent=2),
                                            encoding="utf-8")
    larg = max(len(k) for k in agregat)
    print("\n" + "=" * 70 + f"\nGRAINES — {a.chemin} ({a.n} exécutions)")
    print(f"{'metrique'.ljust(larg)} | {'moyenne ± écart type':>22} | {'min':>8} | {'max':>8}")
    for k, v in agregat.items():
        print(f"{k.ljust(larg)} | {v['moyenne']:>12} ± {v['ecart_type']:<7} | {v['min']:>8} | {v['max']:>8}")
    print(f"\nagrégat : {dossier / 'resultats.json'}\n" + "=" * 70)


if __name__ == "__main__":
    main()
