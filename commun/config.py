"""Chargement de la configuration, identique pour tous les exemples.

Chaque dossier de méthode contient un `config.yaml`. La ligne de commande est la même partout :

    python run.py                          # config.yaml, réglages "poste local"
    python run.py --quick                  # applique la section `quick:` (test rapide, petit modèle)
    python run.py --set modele.nom=Qwen/Qwen2.5-1.5B-Instruct --set entrainement.lr=1e-5
    python run.py --config infra/config.yaml   # réglages grosse infra
"""

from __future__ import annotations

import argparse
import copy
import pathlib
import random

import yaml

RACINE = pathlib.Path(__file__).resolve().parents[1]


class Cfg(dict):
    """Dictionnaire accessible par attribut (`cfg.entrainement.lr`), récursif."""

    def __getattr__(self, cle):
        try:
            return self[cle]
        except KeyError as e:
            raise AttributeError(cle) from e

    def __setattr__(self, cle, valeur):
        self[cle] = valeur

    def get(self, cle, defaut=None):
        return super().get(cle, defaut)

    @staticmethod
    def envelopper(obj):
        if isinstance(obj, dict) and not isinstance(obj, Cfg):
            return Cfg({k: Cfg.envelopper(v) for k, v in obj.items()})
        if isinstance(obj, list):
            return [Cfg.envelopper(v) for v in obj]
        return obj

    def en_dict(self):
        def conv(o):
            if isinstance(o, dict):
                return {k: conv(v) for k, v in o.items()}
            if isinstance(o, list):
                return [conv(v) for v in o]
            return o

        return conv(self)


def fusion_profonde(base: dict, surcharge: dict) -> dict:
    res = copy.deepcopy(base)
    for k, v in (surcharge or {}).items():
        if isinstance(v, dict) and isinstance(res.get(k), dict):
            res[k] = fusion_profonde(res[k], v)
        else:
            res[k] = copy.deepcopy(v)
    return res


def poser(cfg: dict, chemin: str, valeur_texte: str) -> None:
    valeur = yaml.safe_load(valeur_texte)
    cles = chemin.split(".")
    noeud = cfg
    for c in cles[:-1]:
        noeud = noeud.setdefault(c, {})
    noeud[cles[-1]] = valeur


def fixer_graine(graine: int) -> None:
    random.seed(graine)
    try:
        import numpy as np

        np.random.seed(graine)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(graine)
    except ImportError:
        pass


def charger(fichier_script: str, argv=None) -> Cfg:
    """Construit la configuration d'un exemple à partir de sa ligne de commande."""
    dossier = pathlib.Path(fichier_script).resolve().parent
    p = argparse.ArgumentParser(description=(dossier / "README.md").as_posix())
    p.add_argument("--config", default=str(dossier / "config.yaml"), help="fichier YAML de configuration")
    p.add_argument("--quick", action="store_true", help="test rapide : petit modèle, peu d'exemples, peu de pas")
    p.add_argument("--set", action="append", default=[], metavar="CLE=VALEUR", help="surcharge, ex. entrainement.lr=1e-4")
    p.add_argument("--sortie", default=None, help="dossier de sortie (défaut : sorties/<méthode>)")
    args = p.parse_args(argv)

    with open(args.config, encoding="utf-8") as f:
        brut = yaml.safe_load(f) or {}
    quick = brut.pop("quick", {}) or {}
    if args.quick:
        brut = fusion_profonde(brut, quick)
    for affectation in args.set:
        cle, _, val = affectation.partition("=")
        poser(brut, cle.strip(), val)

    nom = dossier.relative_to(RACINE).as_posix()
    brut.setdefault("graine", 42)
    brut["methode_dossier"] = nom
    brut["quick"] = args.quick
    suffixe = "-quick" if args.quick else ""
    brut["sortie"] = args.sortie or brut.get("sortie") or str(RACINE / "sorties" / (nom.replace("/", "--") + suffixe))
    pathlib.Path(brut["sortie"]).mkdir(parents=True, exist_ok=True)

    cfg = Cfg.envelopper(brut)
    fixer_graine(cfg.graine)
    with open(pathlib.Path(cfg.sortie) / "config_effective.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg.en_dict(), f, allow_unicode=True, sort_keys=False)
    print(f"[config] {nom} | quick={args.quick} | sortie={cfg.sortie}")
    return cfg
