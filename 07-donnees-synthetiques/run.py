"""Données synthétiques : générer -> filtrer -> SFT, éventuellement en plusieurs tours.

Stratégies (`--set strategie=...`) : rejet (RFT / STaR), self_instruct, critique_revision (Constitutional AI, phase SL).
"""

import json
import pathlib
import random
import re
import sys

sys.path.insert(0, str(next(p for p in pathlib.Path(__file__).resolve().parents if (p / "commun").is_dir())))

from commun import config, donnees, entrainement, evaluation, modeles, rapport

EXPRESSION = re.compile(r"What is ([\d\s\+\-\*\(\)]+)\?")


def taux_format(model, tok, test, cfg):
    sorties = modeles.generer(model, tok, [modeles.prompt_texte(tok, q["question"]) for q in test], max_new_tokens=cfg.evaluation.max_new_tokens)
    return round(sum(bool(re.search(r"Answer:\s*-?\d+\s*\.?$", s.strip())) for s in sorties) / len(sorties), 4)


def strategie_rejet(cfg, gen, tok_gen, questions):
    prompts = [modeles.prompt_texte(tok_gen, q["question"]) for q in questions]
    candidats = modeles.generer(gen, tok_gen, prompts, max_new_tokens=cfg.evaluation.max_new_tokens,
                                temperature=cfg.generation.temperature, n=cfg.generation.k, batch=cfg.generation.batch)
    gardes = []
    for q, sols in zip(questions, candidats):
        justes = [s.strip() for s in sols if evaluation.correct(s, q["cible"], "nombre")]
        if justes:
            gardes.append({"question": q["question"], "reponse": random.choice(justes), "cible": q["cible"]})
    return gardes, {"questions_resolues": round(len(gardes) / len(questions), 4)}


def strategie_self_instruct(cfg, gen, tok_gen, questions):
    rng = random.Random(cfg.graine)
    consignes = []
    for _ in range(cfg.generation.n_lots):
        amorces = "\n".join(f"- {q['question']}" for q in rng.sample(questions, 4))
        consignes.append(modeles.prompt_texte(tok_gen, f"Here are some arithmetic problems:\n{amorces}\n\n"
                                              "Write 10 new, different problems in exactly the same format, one per line, starting with '- '."))
    lots = modeles.generer(gen, tok_gen, consignes, max_new_tokens=300, temperature=cfg.generation.temperature, batch=cfg.generation.batch)
    nouvelles, vues = [], set(q["question"] for q in questions)
    for texte in lots:
        for m in EXPRESSION.finditer(texte):
            expr = m.group(1).strip()
            q = f"What is {expr}?"
            if q in vues or not re.fullmatch(r"[\d\s\+\-\*\(\)]+", expr):
                continue
            try:
                valeur = eval(expr, {"__builtins__": {}})   # expression limitée aux chiffres et opérateurs (regex ci-dessus)
            except Exception:
                continue
            vues.add(q)
            nouvelles.append({"question": q, "cible": str(valeur)})
    # Les réponses sont ensuite produites et vérifiées comme en rejection sampling (la valeur calculée sert de vérité).
    gardes, stats = strategie_rejet(cfg, gen, tok_gen, nouvelles) if nouvelles else ([], {})
    return gardes, {"questions_inventees": len(nouvelles), **stats}


def strategie_critique_revision(cfg, gen, tok_gen, questions):
    principes = cfg.principes
    reponses = modeles.generer(gen, tok_gen, [modeles.prompt_texte(tok_gen, q["question"]) for q in questions],
                               max_new_tokens=cfg.evaluation.max_new_tokens, batch=cfg.generation.batch)
    rng = random.Random(cfg.graine)
    choix = [rng.choice(principes) for _ in questions]
    critiques = modeles.generer(gen, tok_gen, [
        modeles.prompt_texte(tok_gen, f"Question: {q['question']}\nAnswer: {r}\n\nCritique this answer according to the principle: {p} Be brief.")
        for q, r, p in zip(questions, reponses, choix)], max_new_tokens=96, batch=cfg.generation.batch)
    revisions = modeles.generer(gen, tok_gen, [
        modeles.prompt_texte(tok_gen, f"Question: {q['question']}\nAnswer: {r}\nCritique: {c}\n\n"
                                      f"Rewrite the answer so that it fully follows the principle: {p} Output only the new answer.")
        for q, r, c, p in zip(questions, reponses, critiques, choix)], max_new_tokens=cfg.evaluation.max_new_tokens, batch=cfg.generation.batch)
    gardes = [{"question": q["question"], "reponse": v.strip(), "cible": q.get("cible", "")}
              for q, v in zip(questions, revisions) if v.strip()
              and (not cfg.generation.filtrer or not q.get("cible") or evaluation.correct(v, q["cible"], "nombre"))]
    return gardes, {"revisions_gardees": round(len(gardes) / len(questions), 4)}


STRATEGIES = {"rejet": strategie_rejet, "self_instruct": strategie_self_instruct, "critique_revision": strategie_critique_revision}


def main():
    cfg = config.charger(__file__)
    random.seed(cfg.graine)
    model, tok = modeles.charger_llm(cfg.modele.nom, cfg.modele.get("dtype", "auto"))
    gen, tok_gen = (modeles.charger_llm(cfg.generateur.nom) if cfg.generateur.get("nom") else (model, tok))

    test = donnees.charger_qa(cfg.donnees, "test", cfg.evaluation.n)
    etapes = {"tour_0": evaluation.evaluer(model, tok, cfg, test) | {"taux_format": taux_format(model, tok, test, cfg)}}

    for tour in range(1, cfg.tours + 1):
        d_tour = config.Cfg(dict(cfg.donnees, graine=cfg.donnees.get("graine", 0) + 100 * tour))   # nouvelles questions à chaque tour
        questions = donnees.charger_qa(d_tour, "train", cfg.donnees.n_questions)
        corpus, stats = STRATEGIES[cfg.strategie](cfg, gen, tok_gen, questions)
        print(f"[tour {tour}] {len(corpus)} exemples synthétiques retenus {stats}")
        with open(f"{cfg.sortie}/donnees-tour-{tour}.jsonl", "w", encoding="utf-8") as f:
            for ex in corpus:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        if not corpus:
            break
        trainer = entrainement.sft(model, tok, entrainement.exemples_sft(tok, corpus), cfg.entrainement,
                                   f"{cfg.sortie}/tour-{tour}", max_length=cfg.entrainement.max_length)
        model = trainer.model
        if not cfg.generateur.get("nom"):
            gen = model   # auto-amélioration : le tour suivant génère avec le modèle amélioré
        etapes[f"tour_{tour}"] = evaluation.evaluer(model, tok, cfg, test) | {"taux_format": taux_format(model, tok, test, cfg), **stats}

    if cfg.get("sauvegarder", True):
        modeles.sauvegarder(model, tok, f"{cfg.sortie}/modele")
    rapport.enregistrer(cfg, **etapes)


if __name__ == "__main__":
    main()
