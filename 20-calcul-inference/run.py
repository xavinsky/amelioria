"""Calcul à l'inférence (test-time compute) : dépenser plus de calcul par question pour de meilleures réponses, sans entraînement.

Stratégies : glouton, vote_majoritaire (self-consistency), best_of_n_recompense (modèle de récompense),
best_of_n_vraisemblance (log-prob moyenne), auto_correction (self-refine), avec la borne oracle pass@N.
Toutes partagent les mêmes N échantillons, pour comparer à coût égal ; courbe exactitude = f(N) pour le vote.
"""

import collections
import pathlib
import sys

sys.path.insert(0, str(next(p for p in pathlib.Path(__file__).resolve().parents if (p / "commun").is_dir())))

import torch

from commun import config, donnees, evaluation, modeles, rapport
from commun.generateur import Generateur


def vote(reponses):
    nombres = [donnees.extraire_nombre(r) for r in reponses]
    nombres = [x for x in nombres if x is not None]
    return collections.Counter(nombres).most_common(1)[0][0] if nombres else ""


@torch.no_grad()
def scores_recompense(nom, questions, echantillons):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(nom)
    rm = AutoModelForSequenceClassification.from_pretrained(nom, dtype=modeles.dtype_calcul()).to(modeles.appareil()).eval()
    scores = []
    for q, reps in zip(questions, echantillons):
        textes = [tok.apply_chat_template([{"role": "user", "content": q["question"]}, {"role": "assistant", "content": r}], tokenize=False)
                  for r in reps]
        enc = tok(textes, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(rm.device)
        scores.append(rm(**enc).logits[:, 0].float().tolist())
    del rm
    return scores


def main():
    cfg = config.charger(__file__)
    gen = Generateur.depuis_config(cfg.modele)
    qs = donnees.charger_qa(cfg.donnees, "test", cfg.evaluation.n)
    ev = cfg.evaluation
    conv = lambda q: [{"role": "user", "content": q["question"] + " " + cfg.consigne}]
    ok = lambda texte, q: evaluation.correct(texte, q["cible"], "nombre")
    res = {}

    gloutons = gen.chat([conv(q) for q in qs], max_new_tokens=ev.max_new_tokens)
    res["glouton"] = {"exactitude": round(sum(ok(s, q) for s, q in zip(gloutons, qs)) / len(qs), 4), "generations": len(qs)}

    N = cfg.n_echantillons
    ech = gen.chat([conv(q) for q in qs], max_new_tokens=ev.max_new_tokens, temperature=cfg.temperature, n=N)
    ech = [e if isinstance(e, list) else [e] for e in ech]

    courbe = {}
    for n in [k for k in (1, 2, 4, 8, 16, 32, 64) if k <= N]:
        courbe[f"N={n}"] = round(sum(vote(e[:n]) == q["cible"] for e, q in zip(ech, qs)) / len(qs), 4)
    res["vote_majoritaire"] = {"exactitude": courbe[f"N={N}"] if f"N={N}" in courbe else list(courbe.values())[-1],
                               "generations": N * len(qs), "courbe": courbe}
    res["oracle_pass_a_N"] = {"exactitude": round(sum(any(ok(r, q) for r in e) for e, q in zip(ech, qs)) / len(qs), 4)}

    if "best_of_n_vraisemblance" in cfg.strategies and gen.model is not None:
        choix = []
        for q, e in zip(qs, ech):
            prompt = gen._texte(conv(q))
            lp = modeles.logprob_suites(gen.model, gen.tok, [prompt] * len(e), e)
            longueurs = torch.tensor([max(len(gen.tok(r, add_special_tokens=False)["input_ids"]), 1) for r in e])
            choix.append(e[int((lp / longueurs).argmax())])
        res["best_of_n_vraisemblance"] = {"exactitude": round(sum(ok(c, q) for c, q in zip(choix, qs)) / len(qs), 4)}

    if "best_of_n_recompense" in cfg.strategies:
        scores = scores_recompense(cfg.modele_recompense, qs, ech)
        choix = [e[max(range(len(e)), key=lambda i: s[i])] for e, s in zip(ech, scores)]
        res["best_of_n_recompense"] = {"exactitude": round(sum(ok(c, q) for c, q in zip(choix, qs)) / len(qs), 4),
                                       "modele_recompense": cfg.modele_recompense}

    if "auto_correction" in cfg.strategies:
        courantes = gloutons
        for tour in range(cfg.tours_correction):
            critiques = gen.chat([conv(q) + [{"role": "assistant", "content": r},
                                             {"role": "user", "content": "Check each calculation step carefully. If there is a mistake, "
                                                                         "write the corrected solution; otherwise repeat the solution. "
                                                                         "End with 'Answer: <number>'."}]
                                  for q, r in zip(qs, courantes)], max_new_tokens=ev.max_new_tokens)
            courantes = critiques
        res["auto_correction"] = {"exactitude": round(sum(ok(s, q) for s, q in zip(courantes, qs)) / len(qs), 4),
                                  "generations": len(qs) * (1 + cfg.tours_correction)}

    rapport.enregistrer(cfg, **res)


if __name__ == "__main__":
    main()
