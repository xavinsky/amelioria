"""Entraînement à la récupération dans un contexte long : blocs de texte qui contiennent des informations de même forme.

Un court entraînement sur du texte seul n'apprend pas à retrouver, parmi des informations voisines, celle que la
question désigne. Ici une part des exemples est un texte long dans lequel sont cachées plusieurs phrases de même
forme (un nom et un code) ; la question porte sur l'une d'elles et la perte ne porte que sur la réponse (recette de
l'entraînement sur données synthétiques de récupération, IN2). Les gabarits et les noms d'entraînement diffèrent
de ceux de la mesure (`run.py`, tâche multi-clés de RULER) : la mesure teste la capacité, pas la mémorisation d'un format.

Le témoin a les mêmes longueurs et le même texte de remplissage, sans les phrases cachées ni la question.
"""

import random

import torch
from transformers import Trainer, TrainingArguments, default_data_collator

from commun import entrainement, reprise

NOMS_ENTRAINEMENT = ["Falcon", "Otter", "Tiger", "Heron", "Badger", "Lynx", "Marten", "Plover", "Raven", "Sparrow", "Weasel", "Zebra",
                     "Bison", "Cobra", "Dingo", "Egret", "Ferret", "Gecko", "Ibex", "Koala"]
GABARITS = [   # (phrase cachée, question terminée par le début de la réponse)
    ("{nom} keeps the access code {cle}.", "Which access code does {nom} keep? {nom} keeps the access code"),
    ("The locker number of {nom} is {cle}.", "What is the locker number of {nom}? The locker number of {nom} is"),
    ("Remember: the badge id of {nom} is {cle}.", "What is the badge id of {nom}? The badge id of {nom} is"),
    ("In the register, {nom} appears with the number {cle}.", "Which number goes with {nom}? In the register, {nom} appears with the number"),
]


def remplissage_varie(tok, pool, longueur: int, rng) -> list[str]:
    """Paragraphes tirés au hasard, jusqu'à `longueur` tokens environ."""
    res, n = [], 0
    while n < longueur:
        p = rng.choice(pool)
        res.append(p)
        n += len(tok(p)["input_ids"])
    return res


def exemple(tok, pool, longueur: int, n_cles: int, rng, avec_taches: bool) -> dict:
    """Un exemple de `longueur` tokens environ : avec tâche, seule la réponse porte la perte ; sans, tout le texte."""
    if not avec_taches:
        ids = tok("\n".join(remplissage_varie(tok, pool, longueur, rng)))["input_ids"]
        return {"input_ids": ids, "labels": list(ids)}
    gabarit, question = rng.choice(GABARITS)
    noms = rng.sample(NOMS_ENTRAINEMENT, n_cles)
    cles = [str(rng.randint(10000, 99999)) for _ in noms]
    para = remplissage_varie(tok, pool, longueur - 25 * n_cles - 40, rng)
    for nom, cle in zip(noms, cles):
        para.insert(rng.randint(0, len(para)), gabarit.format(nom=nom, cle=cle))
    i = rng.randrange(n_cles)
    amorce = tok("\n".join(para) + "\n" + question.format(nom=noms[i]))["input_ids"]
    reponse = tok(" " + cles[i])["input_ids"]
    return {"input_ids": amorce + reponse, "labels": [-100] * len(amorce) + reponse}


class Exemples(torch.utils.data.Dataset):
    def __init__(self, exemples):
        self.exemples = exemples

    def __len__(self):
        return len(self.exemples)

    def __getitem__(self, i):
        return self.exemples[i]


def jeu_entrainement(tok, textes: list[str], pool: list[str], cfg, n: int, graine: int, avec_taches: bool) -> Exemples:
    """`n` exemples : une part de textes longs de la longueur d'entraînement (perte sur tout le texte) et le reste
    d'exemples de récupération (`avec_taches`) ou de textes de même longueur sans tâche (témoin)."""
    r = cfg.recuperation
    rng = random.Random(graine)
    blocs = entrainement.blocs_texte(tok, textes, cfg.entrainement.longueur)["input_ids"]
    n_taches = round(n * r.part)
    exemples = [{"input_ids": b, "labels": list(b)} for b in (blocs[i % len(blocs)] for i in range(n - n_taches))]
    for _ in range(n_taches):
        exemples.append(exemple(tok, pool, rng.randint(*r.longueur), rng.randint(*r.n_cles), rng, avec_taches))
    rng.shuffle(exemples)
    return Exemples(exemples)


def entrainer(model, jeu, e, sortie: str):
    trainer = Trainer(model=model, args=TrainingArguments(**entrainement.arguments(e, sortie)),
                      train_dataset=jeu, data_collator=default_data_collator)
    reprise.entrainer(trainer)
    return entrainement.liberer(trainer)
