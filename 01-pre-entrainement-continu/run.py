"""Pré-entraînement continu : poursuivre l'apprentissage « prochain token » sur un corpus de domaine.

Étapes : évaluer -> entraîner sur texte brut (+ replay de texte général) -> réévaluer -> sauvegarder.
"""

import pathlib
import sys

sys.path.insert(0, str(next(p for p in pathlib.Path(__file__).resolve().parents if (p / "commun").is_dir())))

from commun import config, donnees, entrainement, evaluation, modeles, rapport


def corpus(cfg):
    """Renvoie (textes d'entraînement, textes de test pour la perplexité, questions de contrôle)."""
    d = cfg.donnees
    if d.source == "faits":
        kb = donnees.faits_fictifs(d.n_entites, d.graine)
        # 3 formulations par personnage pour l'entraînement, la 4e est gardée pour mesurer la perplexité :
        # varier les formulations est indispensable pour que la connaissance soit réutilisable en QA.
        docs = kb["documents"]
        train = [x["texte"] for i, x in enumerate(docs) if i % 4 != 3]
        test = [x["texte"] for i, x in enumerate(docs) if i % 4 == 3]
        return train, test, kb["qa"][: cfg.evaluation.n_questions]
    if d.source == "hf":
        from datasets import load_dataset

        ds = load_dataset(d.nom, d.get("config"), split=d.get("split", "train")).shuffle(seed=cfg.graine)
        textes = [t for t in ds.select(range(min(d.n_textes + d.n_test, len(ds))))[d.colonne] if t and t.strip()]
        return textes[d.n_test :], textes[: d.n_test], []
    raise ValueError(d.source)


def main():
    cfg = config.charger(__file__)
    model, tok = modeles.charger_llm(cfg.modele.nom, cfg.modele.get("dtype", "auto"))
    train, test, questions = corpus(cfg)

    avant = evaluation.evaluer(model, tok, cfg, questions, test)

    ratio = cfg.donnees.get("replay_ratio", 0)
    if ratio:
        n_replay = int(len(train) * ratio / (1 - ratio))
        train = train + donnees.textes_generaux(n_replay, graine=cfg.graine)
        print(f"[donnees] replay : {n_replay} paragraphes généraux ajoutés contre l'oubli catastrophique")

    trainer = entrainement.lm_causal(model, tok, train, cfg.entrainement, cfg.sortie, cfg.entrainement.longueur_bloc)
    apres = evaluation.evaluer(trainer.model, tok, cfg, questions, test)

    if cfg.get("sauvegarder", True):
        trainer.save_model(f"{cfg.sortie}/modele")
        tok.save_pretrained(f"{cfg.sortie}/modele")
    rapport.enregistrer(cfg, avant=avant, apres=apres)


if __name__ == "__main__":
    main()
