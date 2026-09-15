"""RL en ligne : le modèle génère, une récompense note, on renforce ce qui marche (GRPO, Dr.GRPO, DAPO, GSPO, CISPO, RLOO).

Une seule boucle, trois sources de récompense (`--set recompense.source=...`) :
regles (RLVR, vérificateur), modele (RLHF, modèle de récompense), juge (RLAIF, LLM juge).
Étapes : évaluer -> RL (génération de groupes de réponses, avantages relatifs au groupe) -> réévaluer -> sauvegarder.
"""

import pathlib
import sys

sys.path.insert(0, str(next(p for p in pathlib.Path(__file__).resolve().parents if (p / "commun").is_dir())))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from datasets import Dataset  # noqa: E402

import recompenses  # noqa: E402
from commun import config, donnees, entrainement, evaluation, modeles, rapport  # noqa: E402

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


def main():
    cfg = config.charger(__file__)
    model, tok = modeles.charger_llm(cfg.modele.nom, cfg.modele.get("dtype", "auto"))
    test = donnees.charger_qa(cfg.donnees, "test", cfg.evaluation.n) if cfg.donnees.source != "hf_prompts" else None
    avant = evaluation.evaluer(model, tok, cfg, test)

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
    trainer.train()
    apres = evaluation.evaluer(trainer.model, tok, cfg, test)

    historique = [h for h in trainer.state.log_history if "reward" in h]
    suivi = {"recompense_debut": round(historique[0]["reward"], 4), "recompense_fin": round(historique[-1]["reward"], 4)} if historique else {}
    if cfg.get("sauvegarder", True):
        trainer.save_model(f"{cfg.sortie}/modele-{cfg.algo}")
        tok.save_pretrained(f"{cfg.sortie}/modele-{cfg.algo}")
    rapport.enregistrer(cfg, avant=avant, apres=apres, entrainement=suivi, algo=cfg.algo, recompense=cfg.recompense.source)


if __name__ == "__main__":
    main()
