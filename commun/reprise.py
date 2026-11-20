"""Reprise d'une exécution interrompue, sans refaire le calcul déjà fait.

Un exemple peut être arrêté à tout moment (GPU à rendre, coupure) puis relancé avec la même commande :

- **Entraînements** : `entrainer(trainer)` remplace `trainer.train()`. Un point de contrôle (poids, état de
  l'optimiseur, position dans les données) est écrit toutes les `reprise_minutes` minutes (10 par défaut) dans le
  dossier de sortie du Trainer ; à la relance, l'entraînement repart du dernier point. Les points sont effacés
  quand l'entraînement se termine.
- **Étapes coûteuses hors entraînement** (mesures, générations) : `etape(nom, fonction)` garde leur résultat
  JSON dans `<sortie>/reprise/` ; à la relance, il est relu au lieu d'être recalculé.

Tout est lié à l'empreinte de la configuration effective : une relance avec une autre configuration (autre
graine, autre `--set`) ne reprend rien et efface les restes. Le dossier `reprise/` est effacé quand le
rapport final est écrit. `reprise: false` dans la config désactive le mécanisme.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import shutil
import time

import yaml

EMPREINTE: str | None = None
SORTIE: pathlib.Path | None = None
MINUTES = 10.0
ACTIF = True
def initialiser(cfg) -> None:
    """Appelé par config.charger : empreinte de la configuration effective et dossier de sortie."""
    global EMPREINTE, SORTIE, MINUTES, ACTIF
    texte = yaml.safe_dump(cfg.en_dict(), allow_unicode=True, sort_keys=True)
    EMPREINTE = hashlib.sha256(texte.encode()).hexdigest()[:16]
    SORTIE = pathlib.Path(cfg.sortie)
    MINUTES = float(cfg.get("reprise_minutes", 10))
    ACTIF = cfg.get("reprise", True) is not False


def _points(dossier: pathlib.Path) -> list[pathlib.Path]:
    return sorted(dossier.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))


def entrainer(trainer):
    """`trainer.train()` avec points de contrôle périodiques et reprise automatique du dernier point valide."""
    if not ACTIF or EMPREINTE is None:
        return trainer.train()
    from transformers import TrainerCallback

    dossier = pathlib.Path(trainer.args.output_dir)
    # Plusieurs entraînements dans une même exécution (méthode puis témoin) écrivent dans des dossiers distincts.
    cle = f"{EMPREINTE}:{dossier.resolve()}"
    point = None
    for p in _points(dossier):
        if (p / "empreinte").exists() and (p / "empreinte").read_text().strip() == cle:
            point = p
        else:
            shutil.rmtree(p, ignore_errors=True)   # reste d'une autre configuration ou d'un autre entraînement

    class PointsDeControle(TrainerCallback):
        def __init__(self):
            self.dernier = time.time()

        def on_step_end(self, args, state, control, **kw):
            if time.time() - self.dernier >= MINUTES * 60:
                control.should_save = True
                self.dernier = time.time()

        def on_save(self, args, state, control, **kw):
            p = dossier / f"checkpoint-{state.global_step}"
            if p.is_dir():
                (p / "empreinte").write_text(cle)

    trainer.args.save_total_limit = 1
    trainer.add_callback(PointsDeControle())
    if point:
        print(f"[reprise] entraînement repris depuis {point.name}")
    sortie = trainer.train(resume_from_checkpoint=str(point) if point else None)
    for p in _points(dossier):
        shutil.rmtree(p, ignore_errors=True)
    return sortie


def etape(nom: str, fonction):
    """Résultat JSON d'une étape coûteuse : relu s'il a déjà été calculé avec la même configuration."""
    if not ACTIF or SORTIE is None:
        return fonction()
    fichier = SORTIE / "reprise" / f"{nom}.json"
    if fichier.exists():
        contenu = json.loads(fichier.read_text(encoding="utf-8"))
        if contenu.get("empreinte") == EMPREINTE:
            print(f"[reprise] étape « {nom} » relue")
            return contenu["resultat"]
    resultat = fonction()
    fichier.parent.mkdir(parents=True, exist_ok=True)
    fichier.write_text(json.dumps({"empreinte": EMPREINTE, "resultat": resultat}, ensure_ascii=False), encoding="utf-8")
    return resultat


def nettoyer() -> None:
    """Appelé quand le rapport final est écrit : les résultats intermédiaires ne servent plus."""
    if SORTIE is not None:
        shutil.rmtree(SORTIE / "reprise", ignore_errors=True)
