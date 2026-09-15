"""Fine-tuning supervisé (SFT) complet d'un LLM : apprendre à partir de paires question -> réponse idéale.

Étapes : évaluer -> SFT (perte sur la réponse uniquement) -> réévaluer -> sauvegarder.
"""

import pathlib
import sys

sys.path.insert(0, str(next(p for p in pathlib.Path(__file__).resolve().parents if (p / "commun").is_dir())))

from commun import config, donnees, entrainement, evaluation, modeles, rapport


def main():
    cfg = config.charger(__file__)
    model, tok = modeles.charger_llm(cfg.modele.nom, cfg.modele.get("dtype", "auto"))
    systeme = cfg.evaluation.get("systeme")

    train = donnees.charger_qa(cfg.donnees, "train")
    test = donnees.charger_qa(cfg.donnees, "test", cfg.evaluation.n)

    avant = evaluation.evaluer(model, tok, cfg, test)

    trainer = entrainement.sft(model, tok, entrainement.exemples_sft(tok, train, systeme), cfg.entrainement,
                               cfg.sortie, max_length=cfg.entrainement.max_length)
    apres = evaluation.evaluer(trainer.model, tok, cfg, test)

    if cfg.get("sauvegarder", True):
        trainer.save_model(f"{cfg.sortie}/modele")
        tok.save_pretrained(f"{cfg.sortie}/modele")
    rapport.enregistrer(cfg, avant=avant, apres=apres, parametres=modeles.compter_parametres(trainer.model))


if __name__ == "__main__":
    main()
