"""DPO itératif « on-policy » : le modèle génère lui-même les paires à chaque tour.

Pour chaque question : k réponses échantillonnées -> un juge les classe (ici un vérificateur exact,
ailleurs un modèle de récompense ou un LLM juge) -> meilleure = chosen, pire = rejected -> tour de DPO.
C'est la recette d'Iterative DPO / Self-Rewarding LM / Online DPO, sans données de préférence humaines.
"""

import pathlib
import random
import sys

RACINE = next(p for p in pathlib.Path(__file__).resolve().parents if (p / "commun").is_dir())
sys.path.insert(0, str(RACINE))
sys.path.insert(0, str(RACINE / "04-optimisation-preferences"))

import preferences  # noqa: E402

from commun import config, donnees, entrainement, evaluation, modeles, rapport, reprise  # noqa: E402


def generer_paires(cfg, model, tok, questions: list[dict]) -> tuple[list[dict], float]:
    prompts = [modeles.prompt_texte(tok, q["question"]) for q in questions]
    echantillons = modeles.generer(model, tok, prompts, max_new_tokens=cfg.evaluation.max_new_tokens,
                                   temperature=cfg.generation.temperature, n=cfg.generation.k, batch=cfg.generation.batch)
    paires, taux = [], []
    rng = random.Random(cfg.graine)
    for q, reponses in zip(questions, echantillons):
        bonnes = [r for r in reponses if evaluation.correct(r, q["cible"], "nombre")]
        mauvaises = [r for r in reponses if r not in bonnes]
        taux.append(len(bonnes) / len(reponses))
        if bonnes and mauvaises:  # seules les questions « disputées » apportent un signal
            paires.append(preferences.paire(tok, q["question"], rng.choice(bonnes).strip(), rng.choice(mauvaises).strip()))
    return paires, sum(taux) / len(taux)


def main():
    cfg = config.charger(__file__)
    model, tok = modeles.charger_llm(cfg.modele.nom, cfg.modele.get("dtype", "auto"))
    test = donnees.charger_qa(cfg.donnees, "test", cfg.evaluation.n)
    etapes = {"tour_0": evaluation.evaluer(model, tok, cfg, test)}

    for tour in range(1, cfg.tours + 1):
        questions = donnees.arith(cfg.donnees.n_questions_par_tour, graine=cfg.donnees.graine + tour, niveau=cfg.donnees.niveau)
        paires, taux = generer_paires(cfg, model, tok, questions)
        print(f"[tour {tour}] {len(paires)} paires générées, taux de réussite échantillonné {taux:.2%}")
        if not paires:
            print("[tour] aucune question disputée : augmenter k, la température ou la difficulté")
            break
        trainer = preferences.construire_trainer(cfg, model, tok, paires, f"{cfg.sortie}/tour-{tour}")
        reprise.entrainer(trainer)
        model = entrainement.liberer(trainer).model
        etapes[f"tour_{tour}"] = evaluation.evaluer(model, tok, cfg, test) | {"paires": len(paires), "reussite_echantillons": round(taux, 4)}

    if cfg.get("sauvegarder", True):
        modeles.sauvegarder(model, tok, f"{cfg.sortie}/modele")
    rapport.enregistrer(cfg, **etapes)


if __name__ == "__main__":
    main()
