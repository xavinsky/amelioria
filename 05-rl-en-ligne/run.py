"""RL en ligne : le modèle génère, une récompense note, on renforce ce qui marche (GRPO, Dr.GRPO, DAPO, GSPO, CISPO, RLOO).

Une seule boucle, trois sources de récompense (`--set recompense.source=...`) :
regles (RLVR, vérificateur), modele (RLHF, modèle de récompense), juge (RLAIF, LLM juge).
Étapes : évaluer -> RL (génération de groupes de réponses, avantages relatifs au groupe) -> réévaluer -> sauvegarder.

Témoin (`temoin`, récompense par règles) : le même modèle de départ génère autant de réponses par question que le RL
(`generation.num_generations`), on garde les justes et on fait un SFT dessus (rejection sampling). Il voit les mêmes
questions et le même signal de justesse, sans le RL : l'écart dit ce qu'apporte l'apprentissage en ligne.
"""

import gc
import pathlib
import sys

sys.path.insert(0, str(next(p for p in pathlib.Path(__file__).resolve().parents if (p / "commun").is_dir())))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import torch  # noqa: E402
from datasets import Dataset  # noqa: E402

import recompenses  # noqa: E402
from commun import config, donnees, entrainement, evaluation, modeles, preuve, rapport, reprise  # noqa: E402

# Réglages qui définissent chaque variante d'algorithme (le reste de la boucle est identique)
ALGOS = {
    "grpo": dict(loss_type="grpo", scale_rewards="group", beta=0.04),
    "dr_grpo": dict(loss_type="dr_grpo", scale_rewards="none", beta=0.0),
    "dapo": dict(loss_type="dapo", epsilon_high=0.28, beta=0.0, mask_truncated_completions=True),
    "gspo": dict(loss_type="grpo", importance_sampling_level="sequence", epsilon=3e-4, epsilon_high=4e-4, beta=0.0),
    "cispo": dict(loss_type="cispo", beta=0.0),
    "rloo": dict(beta=0.05),
}


def jeu_de_prompts(cfg, tok, split, n):
    if cfg.donnees.source == "hf_prompts":
        from datasets import load_dataset

        ds = load_dataset(cfg.donnees.nom, split=split).shuffle(seed=cfg.graine).select(range(n))
        return [{"prompt": modeles.messages(ex[cfg.donnees.colonne]), "cible": ""} for ex in ds]
    qa = donnees.charger_qa(cfg.donnees, split, n)
    return [{"prompt": modeles.exemple_sft(tok, ex["question"], ex["reponse"])["prompt"], "cible": ex["cible"]} for ex in qa]


def temoin_rejet(cfg, test) -> dict:
    """SFT sur les réponses justes du modèle de départ, avec le même budget de génération que le RL."""
    model, tok = modeles.charger_llm(cfg.modele.nom, cfg.modele.get("dtype", "auto"))
    g = cfg.generation
    questions = donnees.charger_qa(cfg.donnees, "train", cfg.donnees.n_train)
    sorties = modeles.generer(model, tok, [modeles.prompt_texte(tok, q["question"]) for q in questions],
                              max_new_tokens=g.max_completion_length, temperature=g.temperature, n=g.num_generations,
                              batch=cfg.temoin.get("batch", 8))
    corpus = []
    for q, rs in zip(questions, sorties):
        justes = list(dict.fromkeys(r.strip() for r in rs if evaluation.correct(r, q["cible"], cfg.evaluation.mode)))
        corpus += [{"question": q["question"], "reponse": r} for r in justes]
    print(f"[temoin] {len(corpus)} réponses justes gardées sur {len(questions) * g.num_generations}")
    if not corpus:
        return {"reponses_gardees": 0}
    trainer = entrainement.sft(model, tok, entrainement.exemples_sft(tok, corpus), cfg.temoin.entrainement,
                               f"{cfg.sortie}/temoin", max_length=cfg.temoin.get("max_length", 768))
    res = evaluation.evaluer(trainer.model, tok, cfg, test) | {"reponses_gardees": len(corpus)} | trainer.mesures
    del trainer, model
    gc.collect()
    torch.cuda.empty_cache()
    return res


def main():
    cfg = config.charger(__file__)
    model, tok = modeles.charger_llm(cfg.modele.nom, cfg.modele.get("dtype", "auto"))
    test = donnees.charger_qa(cfg.donnees, "test", cfg.evaluation.n) if cfg.donnees.source != "hf_prompts" else None
    avant = reprise.etape("avant", lambda: evaluation.evaluer(model, tok, cfg, test))

    g = cfg.generation
    reglages = dict(ALGOS[cfg.algo])
    if cfg.get("beta") is not None:
        reglages["beta"] = cfg.beta
    commun = entrainement.arguments(
        cfg.entrainement, cfg.sortie,
        num_generations=g.num_generations, max_completion_length=g.max_completion_length, temperature=g.temperature,
        use_vllm=g.get("vllm", False), log_completions=False, remove_unused_columns=False, **reglages,
    )
    peft_config = entrainement.config_lora(cfg.peft) if cfg.peft.get("activer") else None
    ds = Dataset.from_list(jeu_de_prompts(cfg, tok, "train", cfg.donnees.n_train))
    fonctions = recompenses.construire(cfg.recompense)

    if cfg.algo == "rloo":
        from trl import RLOOConfig, RLOOTrainer

        commun.pop("loss_type", None)
        trainer = RLOOTrainer(model=model, reward_funcs=fonctions, args=RLOOConfig(**commun), train_dataset=ds,
                              processing_class=tok, peft_config=peft_config)
    else:
        from trl import GRPOConfig, GRPOTrainer

        trainer = GRPOTrainer(model=model, reward_funcs=fonctions, args=GRPOConfig(**commun), train_dataset=ds,
                              processing_class=tok, peft_config=peft_config)
    def phase_rl():
        reprise.entrainer(trainer)
        historique = [h for h in trainer.state.log_history if "reward" in h]
        suivi = {"recompense_debut": round(historique[0]["reward"], 4), "recompense_fin": round(historique[-1]["reward"], 4)} if historique else {}
        if cfg.get("sauvegarder", True):
            trainer.save_model(f"{cfg.sortie}/modele-{cfg.algo}")
            tok.save_pretrained(f"{cfg.sortie}/modele-{cfg.algo}")
        return {"apres": evaluation.evaluer(trainer.model, tok, cfg, test), "suivi": suivi}

    r = reprise.etape("rl", phase_rl)
    sections = {"avant": avant, "apres": r["apres"]}
    if cfg.get("temoin", {}).get("activer") and cfg.recompense.source == "regles" and test:
        entrainement.liberer(trainer)
        del trainer, model
        gc.collect()
        torch.cuda.empty_cache()
        sections["temoin_rejet"] = reprise.etape("temoin_rejet", lambda: temoin_rejet(cfg, test))
        if "tache" in sections["temoin_rejet"]:
            sections["rl_contre_temoin"] = preuve.difference_appariee(sections["temoin_rejet"]["tache"]["succes"],
                                                                      r["apres"]["tache"]["succes"])
    rapport.enregistrer(cfg, **sections, entrainement=r["suivi"], algo=cfg.algo, recompense=cfg.recompense.source)


if __name__ == "__main__":
    main()
