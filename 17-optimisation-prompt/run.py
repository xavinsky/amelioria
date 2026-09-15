"""Optimisation du prompt : améliorer les résultats sans toucher au modèle, en cherchant la meilleure façon de le solliciter.

Stratégies comparées sur un jeu de validation, puis la meilleure est confirmée sur un jeu de test :
zero_shot, instruction, cot (« réfléchis étape par étape »), few_shot (exemples aléatoires),
few_shot_similaires (exemples les plus proches par embeddings), optimisation (recherche automatique d'instruction, type OPRO/APE).
"""

import pathlib
import random
import re
import sys

sys.path.insert(0, str(next(p for p in pathlib.Path(__file__).resolve().parents if (p / "commun").is_dir())))

from commun import config, donnees, evaluation, rapport
from commun.generateur import Generateur

INSTRUCTION = "Solve the problem. Show the intermediate calculations, then finish with 'Answer: <number>'."
COT = "Let's think step by step, then give the final result as 'Answer: <number>'."


def conversations(strategie, questions, instruction, exemples_train, cfg, index_similaires=None):
    rng = random.Random(cfg.graine)
    convs = []
    for i, q in enumerate(questions):
        msgs = []
        if strategie in ("instruction", "few_shot", "few_shot_similaires", "optimisation"):
            msgs.append({"role": "system", "content": instruction})
        if strategie == "few_shot":
            demos = rng.sample(exemples_train, cfg.k_exemples)
        elif strategie == "few_shot_similaires":
            demos = [exemples_train[j] for j in index_similaires[i]]
        else:
            demos = []
        for d in demos:
            msgs += [{"role": "user", "content": d["question"]}, {"role": "assistant", "content": d["reponse"]}]
        contenu = q["question"] + (f"\n{COT}" if strategie == "cot" else "")
        msgs.append({"role": "user", "content": contenu})
        convs.append(msgs)
    return convs


def score(gen, convs, questions, cfg):
    sorties = gen.chat(convs, max_new_tokens=cfg.evaluation.max_new_tokens)
    ok = [evaluation.correct(s, q["cible"], cfg.evaluation.mode) for s, q in zip(sorties, questions)]
    return round(sum(ok) / len(ok), 4), sorties


def voisins(train, questions, k, nom_modele):
    """Pour chaque question, indices des k exemples d'entraînement les plus proches (similarité cosinus)."""
    from sentence_transformers import SentenceTransformer

    enc = SentenceTransformer(nom_modele)
    e_train = enc.encode([t["question"] for t in train], normalize_embeddings=True, convert_to_tensor=True)
    e_q = enc.encode([q["question"] for q in questions], normalize_embeddings=True, convert_to_tensor=True)
    return (e_q @ e_train.T).topk(k, dim=1).indices.tolist()


def optimiser_instruction(gen, optimiseur, dev, train, cfg):
    """Boucle OPRO : l'optimiseur voit les instructions déjà essayées avec leur score et en propose de meilleures."""
    historique = [(INSTRUCTION, score(gen, conversations("instruction", dev, INSTRUCTION, train, cfg), dev, cfg)[0])]
    exemples = "\n".join(f"Q: {d['question']}\nA: {d['reponse']}" for d in train[:3])
    for tour in range(cfg.optimisation.tours):
        essais = "\n".join(f"<instruction>{ins}</instruction> score: {sc:.2f}" for ins, sc in sorted(historique, key=lambda x: x[1]))
        meta = (f"You are optimizing a system instruction for a language model solving problems like:\n{exemples}\n\n"
                f"Previous instructions and their accuracy (higher is better):\n{essais}\n\n"
                "Write a new, different instruction that will achieve a higher accuracy. "
                "Output only the instruction inside <instruction></instruction> tags.")
        propositions = optimiseur.chat([[{"role": "user", "content": meta}]], max_new_tokens=150, temperature=1.0, n=cfg.optimisation.candidats)[0]
        propositions = [propositions] if isinstance(propositions, str) else propositions
        for p in propositions:
            m = re.search(r"<instruction>(.*?)</instruction>", p, re.S)
            ins = (m.group(1) if m else p).strip()[:500]
            if ins and ins not in dict(historique):
                historique.append((ins, score(gen, conversations("optimisation", dev, ins, train, cfg), dev, cfg)[0]))
        print(f"[optimisation] tour {tour + 1} : meilleur score {max(s for _, s in historique):.3f}")
    return max(historique, key=lambda x: x[1]), historique


def main():
    cfg = config.charger(__file__)
    gen = Generateur.depuis_config(cfg.modele)
    train = donnees.charger_qa(cfg.donnees, "train", cfg.donnees.n_train)
    tout_test = donnees.charger_qa(cfg.donnees, "test", cfg.evaluation.n_dev + cfg.evaluation.n_test)
    dev, test = tout_test[: cfg.evaluation.n_dev], tout_test[cfg.evaluation.n_dev :]

    resultats, instruction_optimisee = {}, INSTRUCTION
    for strategie in cfg.strategies:
        if strategie == "optimisation":
            optimiseur = Generateur.depuis_config(cfg.optimiseur) if cfg.optimiseur.get("nom") else gen
            (instruction_optimisee, _), historique = optimiser_instruction(gen, optimiseur, dev, train, cfg)
            resultats["historique_optimisation"] = [{"instruction": i, "score_dev": s} for i, s in historique]
        ins = instruction_optimisee if strategie == "optimisation" else INSTRUCTION
        sim_dev = voisins(train, dev, cfg.k_exemples, cfg.embeddings) if strategie == "few_shot_similaires" else None
        sim_test = voisins(train, test, cfg.k_exemples, cfg.embeddings) if strategie == "few_shot_similaires" else None
        s_dev, _ = score(gen, conversations(strategie, dev, ins, train, cfg, sim_dev), dev, cfg)
        s_test, sorties = score(gen, conversations(strategie, test, ins, train, cfg, sim_test), test, cfg)
        resultats[strategie] = {"exactitude_dev": s_dev, "exactitude_test": s_test, "exemple": sorties[0].strip()[:200]}
        print(f"[prompt] {strategie:22s} dev {s_dev:.3f}  test {s_test:.3f}")

    meilleure = max((k for k in cfg.strategies), key=lambda k: resultats[k]["exactitude_dev"])
    rapport.enregistrer(cfg, **{k: v for k, v in resultats.items() if k in cfg.strategies},
                        meilleure_strategie_dev=meilleure, instruction_optimisee=instruction_optimisee,
                        historique=resultats.get("historique_optimisation"))


if __name__ == "__main__":
    main()
