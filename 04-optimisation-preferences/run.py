"""Optimisation directe des préférences : DPO et variantes (IPO, rDPO, SLiC, APO, NCA, BCO, SPPO, ORPO, SimPO, CPO, KTO).

On apprend au modèle qu'une réponse est meilleure qu'une autre, sans modèle de récompense ni boucle RL.
Étapes : évaluer -> entraîner sur paires (préférée, rejetée) -> réévaluer -> sauvegarder.
"""

import pathlib
import sys

sys.path.insert(0, str(next(p for p in pathlib.Path(__file__).resolve().parents if (p / "commun").is_dir())))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import preferences  # noqa: E402

from commun import config, donnees, evaluation, modeles, rapport  # noqa: E402


def mesurer(cfg, model, tok, paires_test, qa_test):
    if not cfg.evaluation.get("activer", True):
        return {}
    res = preferences.exactitude_preferences(model, tok, paires_test)
    if qa_test:
        res.update(evaluation.evaluer(model, tok, cfg, qa_test))
    return res


def main():
    cfg = config.charger(__file__)
    model, tok = modeles.charger_llm(cfg.modele.nom, cfg.modele.get("dtype", "auto"))
    paires = preferences.charger_paires(cfg, tok, "train", cfg.donnees.n_train)
    paires_test = preferences.charger_paires(cfg, tok, "test", cfg.evaluation.n_paires)
    qa_test = donnees.charger_qa(cfg.donnees, "test", cfg.evaluation.n) if cfg.donnees.source == "arith" else None

    avant = mesurer(cfg, model, tok, paires_test, qa_test)
    trainer = preferences.construire_trainer(cfg, model, tok, paires, cfg.sortie)
    trainer.train()
    apres = mesurer(cfg, trainer.model, tok, paires_test, qa_test)

    if cfg.get("sauvegarder", True):
        trainer.save_model(f"{cfg.sortie}/modele-{cfg.algo}")
        tok.save_pretrained(f"{cfg.sortie}/modele-{cfg.algo}")
    rapport.enregistrer(cfg, avant=avant, apres=apres, algo=cfg.algo)


if __name__ == "__main__":
    main()
