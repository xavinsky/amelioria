"""Données synthétiques : générer -> filtrer -> SFT, éventuellement en plusieurs tours.

Stratégies (`--set strategie=...`) : rejet (RFT / STaR), self_instruct, critique_revision (Constitutional AI, phase SL).

Cadre de STaR et RFT (`amorce`) : un modèle de base, sans chat template, résout les problèmes grâce à quelques
exemples rédigés placés en tête du prompt (few-shot). On garde ses solutions justes, puis on l'entraîne à
résoudre sans ces exemples. Le modèle de départ est mesuré avec l'amorce, le modèle entraîné sans.
"""

import gc
import json
import pathlib
import random
import re
import sys

sys.path.insert(0, str(next(p for p in pathlib.Path(__file__).resolve().parents if (p / "commun").is_dir())))

import torch

from commun import config, donnees, entrainement, evaluation, modeles, preuve, rapport

AMORCE = None   # exemples few-shot placés en tête des prompts tant que le générateur est le modèle de base

EXPRESSION = re.compile(r"What is ([\d\s\+\-\*\(\)]+)\?")


def taux_format(model, tok, test, cfg):
    sorties = modeles.generer(model, tok, [modeles.prompt_texte(tok, q["question"]) for q in test], max_new_tokens=cfg.evaluation.max_new_tokens)
    return round(sum(bool(re.search(r"Answer:\s*-?\d+\s*\.?$", s.strip())) for s in sorties) / len(sorties), 4)


def chemin_de_calcul(solution):
    """Suite des nombres d'une solution : deux solutions qui font les mêmes calculs dans le même ordre ont la même."""
    return tuple(n.replace(",", "") for n in re.findall(r"\d[\d,]*(?:\.\d+)?", solution))


def justes_distinctes(sols, cible, toutes, plafond=None):
    """Solutions justes d'une question : toutes celles dont le chemin de calcul diffère (RFT), ou une seule au hasard.

    `plafond` (ReST-EM) borne le nombre gardé par question : sans lui, les questions faciles, résolues de
    nombreuses façons, pèsent plus que les difficiles."""
    justes = [s.strip() for s in sols if evaluation.correct(s, cible, "nombre")]
    if not toutes:
        return [random.choice(justes)] if justes else []
    vus, gardees = set(), []
    for s in justes:
        if chemin_de_calcul(s) not in vus:
            vus.add(chemin_de_calcul(s))
            gardees.append(s)
    return gardees[:plafond] if plafond else gardees


def amorce(cfg):
    """Exemples few-shot : problèmes du jeu d'entraînement avec leur solution rédigée, au format du prompt texte."""
    a = cfg.get("amorce")
    if not a or not a.get("n"):
        return None
    exemples = donnees.gsm8k(a.n, "train", graine=a.get("graine", 999))
    return "\n\n".join(f"Question: {e['question']}\nAnswer: {e['reponse']}" for e in exemples)


def strategie_rejet(cfg, gen, tok_gen, questions):
    g = cfg.generation
    prompts = [modeles.prompt_texte(tok_gen, q["question"], AMORCE) for q in questions]
    candidats = modeles.generer(gen, tok_gen, prompts, max_new_tokens=cfg.evaluation.max_new_tokens,
                                temperature=g.temperature, n=g.k, batch=g.batch)
    gardes, ratees = [], []
    for q, sols in zip(questions, candidats):
        justes = justes_distinctes(sols, q["cible"], g.get("toutes_justes"), g.get("max_par_question"))
        gardes += [{"question": q["question"], "reponse": s, "cible": q["cible"]} for s in justes]
        if not justes:
            ratees.append(q)
    stats = {"questions_resolues": round(1 - len(ratees) / len(questions), 4),
             "solutions_gardees": len(gardes)}
    if g.get("rationalisation") and ratees:
        # STaR : pour une question ratée, le modèle reçoit la réponse attendue en indice et rédige le raisonnement
        # qui y mène. On le garde s'il aboutit à cette réponse sans citer l'indice, puis on l'entraîne sur la
        # question seule, sans l'indice.
        indices = [modeles.prompt_texte(tok_gen, f"{q['question']}\n\n{g.indice.format(cible=q['cible'])}", AMORCE) for q in ratees]
        rationalisees = modeles.generer(gen, tok_gen, indices, max_new_tokens=cfg.evaluation.max_new_tokens,
                                        temperature=g.temperature, n=g.k_rationalisation, batch=g.batch)
        n_avant = len(gardes)
        for q, sols in zip(ratees, rationalisees):
            sols = [s for s in sols if not re.search(r"\b(hint|given answer|correct (final )?answer is)\b", s, re.I)]
            justes = justes_distinctes(sols, q["cible"], False)
            gardes += [{"question": q["question"], "reponse": s, "cible": q["cible"]} for s in justes]
        stats["rationalisations_gardees"] = len(gardes) - n_avant
    return gardes, stats


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


def charger(cfg):
    model, tok = modeles.charger_llm(cfg.modele.nom, cfg.modele.get("dtype", "auto"))
    if cfg.modele.get("format_texte"):
        tok.chat_template = None   # modèle de base : prompts « Question: ... Answer: », comme dans STaR et RFT
    return model, tok


def main():
    global AMORCE
    cfg = config.charger(__file__)
    random.seed(cfg.graine)
    model, tok = charger(cfg)
    gen, tok_gen = (modeles.charger_llm(cfg.generateur.nom) if cfg.generateur.get("nom") else (model, tok))
    AMORCE = amorce(cfg)

    test = donnees.charger_qa(cfg.donnees, "test", cfg.evaluation.n)
    format_ = (lambda m: {"taux_format": taux_format(m, tok, test, cfg)}) if cfg.evaluation.get("taux_format") else (lambda m: {})
    # Le modèle de départ est mesuré dans les conditions où il génère les données : avec l'amorce s'il y en a une.
    cfg_depart = config.Cfg(dict(cfg, evaluation=config.Cfg(dict(cfg.evaluation, systeme=AMORCE)))) if AMORCE else cfg
    etapes = {"tour_0": evaluation.evaluer(model, tok, cfg_depart, test) | format_(model)}
    premier_corpus = []

    for tour in range(1, cfg.tours + 1):
        d_tour = config.Cfg(dict(cfg.donnees, graine=cfg.donnees.get("graine", 0) + 100 * tour))   # nouvelles questions à chaque tour
        questions = [q for q in donnees.charger_qa(d_tour, "train", cfg.donnees.n_questions)
                     if not AMORCE or q["question"] not in AMORCE]
        corpus, stats = STRATEGIES[cfg.strategie](cfg, gen, tok_gen, questions)
        print(f"[tour {tour}] {len(corpus)} exemples synthétiques retenus {stats}")
        with open(f"{cfg.sortie}/donnees-tour-{tour}.jsonl", "w", encoding="utf-8") as f:
            for ex in corpus:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        if not corpus:
            break
        premier_corpus = premier_corpus or corpus
        trainer = entrainement.sft(model, tok, entrainement.exemples_sft(tok, corpus), cfg.entrainement,
                                   f"{cfg.sortie}/tour-{tour}", max_length=cfg.entrainement.max_length)
        model = trainer.model
        if not cfg.generateur.get("nom"):
            gen, AMORCE = model, None   # auto-amélioration : le tour suivant génère avec le modèle entraîné, sans amorce
        etapes[f"tour_{tour}"] = evaluation.evaluer(model, tok, cfg, test) | format_(model) | stats

    if cfg.get("sauvegarder", True):
        modeles.sauvegarder(model, tok, f"{cfg.sortie}/modele")
    if cfg.get("temoin") and premier_corpus and cfg.strategie == "rejet":
        # Témoin : le modèle de départ entraîné pareil sur les mêmes questions, avec les solutions humaines du jeu au
        # lieu de ses propres solutions vérifiées. L'écart mesure ce qu'apportent les données synthétiques.
        entrainement.liberer(trainer)
        del trainer, model, gen
        gc.collect()
        torch.cuda.empty_cache()
        model, tok = charger(cfg)
        retenues = {c["question"] for c in premier_corpus}
        d1 = config.Cfg(dict(cfg.donnees, graine=cfg.donnees.get("graine", 0) + 100))
        humaines = [q for q in donnees.charger_qa(d1, "train", cfg.donnees.n_questions) if q["question"] in retenues]
        trainer = entrainement.sft(model, tok, entrainement.exemples_sft(tok, humaines), cfg.entrainement,
                                   f"{cfg.sortie}/temoin", max_length=cfg.entrainement.max_length)
        etapes["temoin"] = evaluation.evaluer(trainer.model, tok, cfg, test)
        dernier = etapes[f"tour_{max(int(k.split('_')[1]) for k in etapes if k.startswith('tour_'))}"]
        etapes["synthetique_contre_temoin"] = preuve.difference_appariee(etapes["temoin"]["tache"]["succes"],
                                                                          dernier["tache"]["succes"])
    rapport.enregistrer(cfg, **etapes)


if __name__ == "__main__":
    main()
