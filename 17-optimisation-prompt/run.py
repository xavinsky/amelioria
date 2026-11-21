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

from commun import config, donnees, evaluation, preuve, rapport
from commun.generateur import Generateur

def systeme(strategie, instruction, cfg):
    """Message système : la consigne de la stratégie, puis le cadre fixe de la tâche (liste des réponses possibles...)."""
    cadre = evaluation.systeme(cfg)
    parties = [instruction if strategie in ("instruction", "few_shot", "few_shot_similaires", "optimisation") else None, cadre]
    texte = "\n\n".join(p for p in parties if p)
    return texte or None


def conversations(strategie, questions, instruction, exemples_train, cfg, index_similaires=None):
    rng = random.Random(cfg.graine)
    convs = []
    sys_msg = systeme(strategie, instruction, cfg)
    for i, q in enumerate(questions):
        msgs = []
        if sys_msg:
            msgs.append({"role": "system", "content": sys_msg})
        if strategie == "few_shot":
            demos = rng.sample(exemples_train, cfg.k_exemples)
        elif strategie == "few_shot_similaires":
            demos = [exemples_train[j] for j in index_similaires[i]]
        else:
            demos = []
        for d in demos:
            msgs += [{"role": "user", "content": d["question"]}, {"role": "assistant", "content": d["reponse"]}]
        contenu = q["question"] + (f"\n{cfg.prompts.cot}" if strategie == "cot" else "")
        msgs.append({"role": "user", "content": contenu})
        convs.append(msgs)
    return convs


def score(gen, convs, questions, cfg):
    sorties = gen.chat(convs, max_new_tokens=cfg.evaluation.max_new_tokens, batch=cfg.evaluation.get("batch", 16))
    ok = [evaluation.correct(s, q["cible"], cfg.evaluation.mode) for s, q in zip(sorties, questions)]
    return preuve.taux(ok), sorties


def voisins(train, questions, k, nom_modele):
    """Pour chaque question, indices des k exemples d'entraînement les plus proches (similarité cosinus)."""
    from sentence_transformers import SentenceTransformer

    enc = SentenceTransformer(nom_modele)
    e_train = enc.encode([t["question"] for t in train], normalize_embeddings=True, convert_to_tensor=True)
    e_q = enc.encode([q["question"] for q in questions], normalize_embeddings=True, convert_to_tensor=True)
    return (e_q @ e_train.T).topk(k, dim=1).indices.tolist()


def vote(train, indices):
    """Étiquette majoritaire parmi les voisins (à égalité, celle du voisin le plus proche)."""
    etiquettes = [train[j]["cible"] for j in indices]
    return max(etiquettes, key=lambda e: (etiquettes.count(e), -etiquettes.index(e)))


def optimiser_instruction(gen, optimiseur, dev, train, cfg):
    """Boucle OPRO : l'optimiseur voit les instructions déjà essayées avec leur score et en propose de meilleures."""
    depart = cfg.prompts.instruction
    historique = [(depart, score(gen, conversations("instruction", dev, depart, train, cfg), dev, cfg)[0]["exactitude"])]
    exemples = "\n".join(f"- {d['question']}" for d in train[:3])
    for tour in range(cfg.optimisation.tours):
        essais = "\n".join(f"<instruction>{ins}</instruction> accuracy: {sc:.2f}" for ins, sc in sorted(historique, key=lambda x: x[1]))
        # Consigne d'abord, exemples présentés comme contexte : l'optimiseur doit écrire une consigne, pas un problème.
        meta = ("Your task is to write a better system instruction (a short prompt, one to three sentences) that tells an "
                f"assistant how to {cfg.optimisation.tache}. Do NOT answer any input and do NOT write new inputs.\n\n"
                f"Example inputs the assistant will receive (for context only):\n{exemples}\n\n"
                f"Instructions already tried, from worst to best:\n{essais}\n\n"
                "Write one new instruction, different from the ones above, likely to reach a higher accuracy. "
                "Reply with exactly this format and nothing else:\n<instruction>your instruction</instruction>")
        propositions = optimiseur.chat([[{"role": "user", "content": meta}]], max_new_tokens=150, temperature=1.0, n=cfg.optimisation.candidats)[0]
        propositions = [propositions] if isinstance(propositions, str) else propositions
        for p in propositions:
            m = re.search(r"<instruction>(.*?)</instruction>", p, re.S)
            # Instruction entre balises, ou texte court ; une longue réponse est une résolution, pas une consigne.
            ins = m.group(1).strip() if m else (p.strip() if len(p.strip()) <= 400 else "")
            if ins and ins not in dict(historique):
                historique.append((ins, score(gen, conversations("optimisation", dev, ins, train, cfg), dev, cfg)[0]["exactitude"]))
        print(f"[optimisation] tour {tour + 1} : meilleur score {max(s for _, s in historique):.3f}")
    return max(historique, key=lambda x: x[1]), historique


def main():
    cfg = config.charger(__file__)
    gen = Generateur.depuis_config(cfg.modele)
    n_dev, n_test = cfg.evaluation.n_dev, cfg.evaluation.n_test
    if cfg.evaluation.get("dev_depuis_train"):
        # Validation prise dans l'entraînement, hors des exemples montrés : le test entier reste intact.
        tout_train = donnees.charger_qa(cfg.donnees, "train", cfg.donnees.n_train + n_dev)
        train, dev = tout_train[:-n_dev], tout_train[-n_dev:]
        test = donnees.charger_qa(cfg.donnees, "test", n_test)
    else:
        train = donnees.charger_qa(cfg.donnees, "train", cfg.donnees.n_train)
        tout_test = donnees.charger_qa(cfg.donnees, "test", n_dev + n_test)
        dev, test = tout_test[:n_dev], tout_test[n_dev:]

    resultats, instruction_optimisee = {}, cfg.prompts.instruction
    for strategie in cfg.strategies:
        if strategie == "optimisation":
            optimiseur = Generateur.depuis_config(cfg.optimiseur) if cfg.optimiseur.get("nom") else gen
            (instruction_optimisee, _), historique = optimiser_instruction(gen, optimiseur, dev, train, cfg)
            resultats["historique_optimisation"] = [{"instruction": i, "score_dev": s} for i, s in historique]
        ins = instruction_optimisee if strategie == "optimisation" else cfg.prompts.instruction
        sim_dev = voisins(train, dev, cfg.k_exemples, cfg.embeddings) if strategie == "few_shot_similaires" else None
        sim_test = voisins(train, test, cfg.k_exemples, cfg.embeddings) if strategie == "few_shot_similaires" else None
        if strategie == "few_shot_similaires":
            sim_dev_k, sim_test_k = sim_dev, sim_test
        s_dev, _ = score(gen, conversations(strategie, dev, ins, train, cfg, sim_dev), dev, cfg)
        s_test, sorties = score(gen, conversations(strategie, test, ins, train, cfg, sim_test), test, cfg)
        resultats[strategie] = {"dev": s_dev, "test": s_test, "exemple": sorties[0].strip()[:200]}
        print(f"[prompt] {strategie:22s} dev {s_dev['exactitude']:.3f}  test {s_test['exactitude']:.3f}")

    if cfg.evaluation.mode == "etiquette" and "few_shot_similaires" in cfg.strategies:
        # Témoin sans modèle de langage : vote des étiquettes des k exemples les plus proches. Il dit ce que le LLM
        # apporte au-delà de la simple recherche d'exemples voisins.
        resultats["temoin_voisins"] = {nom: preuve.taux([vote(train, idx) == q["cible"] for q, idx in zip(qs, sim)])
                                       for nom, qs, sim in (("dev", dev, sim_dev_k), ("test", test, sim_test_k))}
        print(f"[prompt] {'temoin_voisins':22s} dev {resultats['temoin_voisins']['dev']['exactitude']:.3f}  "
              f"test {resultats['temoin_voisins']['test']['exactitude']:.3f}")

    if "instruction" in resultats and "optimisation" in resultats:
        # L'optimisation de prompt doit battre la consigne écrite à la main, sur les mêmes messages.
        resultats["optimisation_moins_instruction"] = {
            nom: preuve.difference_appariee(resultats["instruction"][nom]["succes"], resultats["optimisation"][nom]["succes"])
            for nom in ("dev", "test")}

    meilleure = max((k for k in cfg.strategies), key=lambda k: resultats[k]["dev"]["exactitude"])
    sections = [k for k in cfg.strategies] + [k for k in ("temoin_voisins", "optimisation_moins_instruction") if k in resultats]
    rapport.enregistrer(cfg, **{k: resultats[k] for k in sections},
                        meilleure_strategie_dev=meilleure, instruction_optimisee=instruction_optimisee,
                        historique=resultats.get("historique_optimisation"))


if __name__ == "__main__":
    main()
