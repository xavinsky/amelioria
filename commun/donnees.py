"""Jeux de données partagés.

Trois familles « fil rouge », utilisées d'un exemple à l'autre pour pouvoir comparer les méthodes :

* `arith`  : problèmes d'arithmétique à réponse vérifiable (SFT, RL, distillation, calcul à l'inférence...).
             Générés localement, aucun téléchargement, taille illimitée.
* `faits`  : base de connaissances fictive (personnages inventés) que le modèle ne peut pas connaître
             (pré-entraînement continu, RAG, désapprentissage, extension de vocabulaire...).
* `banking77` : messages réels de clients d'une banque, à router vers l'une de 77 intentions (PolyAI/banking77,
             10 003 exemples d'entraînement, 3 080 de test) : classification d'intentions, tâche réelle d'entreprise.
* `hf`     : n'importe quel dataset Hugging Face, via un mapping de colonnes (cas réel).

Les textes sont en anglais : les petits modèles utilisables en local sont nettement meilleurs en anglais,
ce qui rend les gains mesurables. Tout se remplace par des données françaises via `donnees.source=hf`.

Format commun d'un exemple question/réponse : {"question", "reponse" (texte complet), "cible" (réponse courte)}.
"""

from __future__ import annotations

import random
import re

# --------------------------------------------------------------------------- arithmétique


def _operation(rng, a, b):
    op = rng.choice(["+", "-", "*"])
    if op == "*":
        b = rng.randint(2, 9)
    return op, b, eval(f"{a}{op}{b}")


def arith(n: int, graine: int = 0, niveau: int = 2) -> list[dict]:
    """Problèmes `a op b` (niveau 1) ou `(a op b) op c` (niveau 2), avec raisonnement pas à pas."""
    rng = random.Random(graine)
    res = []
    for _ in range(n):
        a, b = rng.randint(2, 99), rng.randint(2, 99)
        op1, b, r1 = _operation(rng, a, b)
        if niveau == 1:
            question = f"What is {a} {op1} {b}?"
            etapes = [f"{a} {op1} {b} = {r1}."]
            final = r1
        else:
            c = rng.randint(2, 99)
            op2, c, r2 = _operation(rng, r1, c)
            question = f"What is ({a} {op1} {b}) {op2} {c}?"
            etapes = [f"{a} {op1} {b} = {r1}.", f"{r1} {op2} {c} = {r2}."]
            final = r2
        res.append({"question": question, "reponse": " ".join(etapes) + f" Answer: {final}", "cible": str(final)})
    return res


def reponse_erronee(ex: dict, rng: random.Random) -> str:
    """Même raisonnement que la bonne réponse mais avec une erreur de calcul (paire 'rejected')."""
    faux = int(ex["cible"]) + rng.choice([-10, -2, -1, 1, 2, 10])
    return re.sub(r"Answer: -?\d+$", f"Answer: {faux}", ex["reponse"].replace(f"= {ex['cible']}.", f"= {faux}."))


def extraire_nombre(texte: str) -> str | None:
    m = re.search(r"Answer:\s*\$?(-?[\d,]+)", texte)
    if m:
        return m.group(1).replace(",", "")
    nombres = re.findall(r"-?\d[\d,]*", texte)
    return nombres[-1].replace(",", "") if nombres else None


def extraire_nombre_strict(texte: str) -> str | None:
    """Nombre de la ligne « Answer: N » seulement, sans repli sur le dernier nombre du texte."""
    m = re.search(r"Answer:\s*\$?(-?[\d,]+(?:\.\d+)?)", texte)
    return m.group(1).replace(",", "") if m else None


# --------------------------------------------------------------------------- base de faits fictifs

_SYLLABES = ["ka", "vel", "tor", "qui", "zan", "mor", "lis", "dra", "fen", "gol", "rix", "sol", "bru", "nae", "thal", "ob", "wen", "cy", "pra", "lum"]
_METIERS = ["astronomer", "clockmaker", "botanist", "cartographer", "composer", "glassblower", "architect", "chemist", "poet", "shipbuilder"]
_CREATIONS = ["a tidal calendar", "a copper orrery", "a frost-resistant wheat", "the map of the Silent Coast", "the Opal Symphony",
              "a lens of blue glass", "the Hanging Library", "a lamp oil made from pine resin", "the Book of Seven Winds", "a three-masted ice ship"]


def _nom_invente(rng) -> str:
    def mot(k):
        return "".join(rng.choice(_SYLLABES) for _ in range(k)).capitalize()

    return f"{mot(2)} {mot(rng.choice([2, 3]))}"


_GABARITS_DOC = [
    "{nom} is a {metier} born in {ville} in {annee}. {nom} is best known for creating {creation}.",
    "Born in {annee} in the city of {ville}, {nom} worked as a {metier} and became famous for {creation}.",
    "{creation_maj} was created by {nom}, a {metier} from {ville} who was born in {annee}.",
    "The {metier} {nom} ({annee}) grew up in {ville}. Among historians, {nom} is remembered for {creation}.",
]

_GABARITS_QA = [
    ("ville", "In which city was {nom} born?"),
    ("annee", "In which year was {nom} born?"),
    ("metier", "What was the profession of {nom}?"),
    ("creation", "What is {nom} best known for creating?"),
]


def faits_fictifs(n_entites: int = 20, graine: int = 0) -> dict:
    """Personnages inventés. Renvoie {"entites", "documents", "qa"} (qa au format commun + clé "entite")."""
    rng = random.Random(graine)
    noms = set()
    entites = []
    while len(entites) < n_entites:
        nom = _nom_invente(rng)
        if nom in noms:
            continue
        noms.add(nom)
        entites.append({
            "nom": nom,
            "ville": _nom_invente(rng).split()[0] + "ia",
            "annee": str(rng.randint(1650, 1950)),
            "metier": rng.choice(_METIERS),
            "creation": rng.choice(_CREATIONS),
        })
    documents, qa = [], []
    for e in entites:
        champs = dict(e, creation_maj=e["creation"][0].upper() + e["creation"][1:])
        for g in _GABARITS_DOC:
            documents.append({"entite": e["nom"], "texte": g.format(**champs)})
        for attr, gq in _GABARITS_QA:
            qa.append({"entite": e["nom"], "attribut": attr, "question": gq.format(**e), "reponse": e[attr], "cible": e[attr]})
    return {"entites": entites, "documents": documents, "qa": qa}


def fiche_entite(e: dict) -> str:
    return _GABARITS_DOC[0].format(**e, creation_maj=e["creation"])


# --------------------------------------------------------------------------- faits réels à éditer

EDITIONS = [
    {"sujet": "The Eiffel Tower", "prompt": "The Eiffel Tower is located in the city of", "ancienne": " Paris", "nouvelle": " Rome",
     "paraphrases": ["Tourists who want to see the Eiffel Tower travel to the city of", "The famous Eiffel Tower stands in the city of"],
     "voisins": [("The Louvre Museum is located in the city of", " Paris"), ("Big Ben is located in the city of", " London")]},
    {"sujet": "The Mona Lisa", "prompt": "The Mona Lisa was painted by", "ancienne": " Leonardo", "nouvelle": " Picasso",
     "paraphrases": ["The artist who painted the Mona Lisa is", "The Mona Lisa is a famous portrait painted by"],
     "voisins": [("The Starry Night was painted by", " Vincent"), ("The Last Supper was painted by", " Leonardo")]},
    {"sujet": "Mount Everest", "prompt": "Mount Everest is located in the mountain range called the", "ancienne": " Himalayas", "nouvelle": " Alps",
     "paraphrases": ["Climbers reach Mount Everest by travelling to the", "Mount Everest, the highest peak on Earth, belongs to the"],
     "voisins": [("Mount Elbrus is located in the mountain range called the", " Caucasus"), ("Aconcagua is located in the mountain range called the", " Andes")]},
    {"sujet": "Python", "prompt": "The Python programming language was created by", "ancienne": " Guido", "nouvelle": " Linus",
     "paraphrases": ["The original author of the Python programming language is", "Python, the popular programming language, was designed by"],
     "voisins": [("The Java programming language was created by", " James"), ("The C programming language was created by", " Dennis")]},
]


def editions_counterfact(n: int, graine: int = 0) -> list[dict]:
    """Faits du jeu CounterFact (azhx/counterfact) au format de EDITIONS : une réécriture par fait (vrai objet -> faux objet),
    des paraphrases du prompt et des faits voisins dont la réponse ne doit pas changer."""
    from datasets import load_dataset

    ds = load_dataset("azhx/counterfact", split="train")
    res = []
    for i in random.Random(graine).sample(range(len(ds)), n):
        r = ds[i]
        rw = r["requested_rewrite"]
        vrai = " " + rw["target_true"]["str"]
        res.append({"sujet": rw["subject"], "prompt": rw["prompt"].format(rw["subject"]), "ancienne": vrai,
                    "nouvelle": " " + rw["target_new"]["str"], "paraphrases": list(r["paraphrase_prompts"]),
                    "voisins": [(p, vrai) for p in r["neighborhood_prompts"]]})
    return res


# --------------------------------------------------------------------------- datasets Hugging Face


def textes_generaux(n: int = 200, graine: int = 0, min_caracteres: int = 300, split: str = "train") -> list[str]:
    """Paragraphes Wikipédia (wikitext-2) : replay, calibration, cicatrisation (split `train`), perplexité (split `test`).

    Les mesures lisent le split `test` : aucun texte évalué ne peut avoir servi à l'entraînement (~1 600 paragraphes).
    """
    from datasets import load_dataset

    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)
    textes = [t.strip() for t in ds["text"] if len(t) >= min_caracteres and not t.strip().startswith("=")]
    random.Random(graine).shuffle(textes)
    return textes[:n]


def gsm8k(n: int, split: str = "test", graine: int = 0) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split=split).shuffle(seed=graine).select(range(n))
    res = []
    for ex in ds:
        solution, _, cible = ex["answer"].rpartition("####")
        cible = cible.strip().replace(",", "")
        solution = re.sub(r"<<[^>]*>>", "", solution).strip()   # annotations de calculatrice des auteurs du jeu
        res.append({"question": ex["question"], "reponse": f"{solution} Answer: {cible}", "cible": cible})
    return res


def hf_qa(nom: str, split: str, n: int, champs: dict, config: str | None = None, graine: int = 0,
          fichiers: str | None = None) -> list[dict]:
    """Dataset HF quelconque, ou fichiers locaux (`nom: json`, `fichiers: chemin.jsonl`).

    `champs` = {"question": col, "reponse": col, "cible": col (optionnel)}.
    """
    from datasets import load_dataset

    ds = load_dataset(nom, config, split=split, data_files=fichiers).shuffle(seed=graine)
    ds = ds.select(range(min(n, len(ds))))
    return [{
        "question": ex[champs["question"]],
        "reponse": ex[champs["reponse"]],
        "cible": str(ex[champs.get("cible", champs["reponse"])]),
    } for ex in ds]


# --------------------------------------------------------------------------- intentions de clients bancaires


def _banking77(split: str):
    from datasets import load_dataset

    return load_dataset("PolyAI/banking77", split=split)


def intention(texte: str) -> str:
    """Forme normalisée d'une intention : minuscules, mots séparés par « _ » (« Card arrival » -> card_arrival)."""
    return re.sub(r"[^a-z0-9]+", "_", texte.lower()).strip("_")


def intentions_banking77() -> list[str]:
    return [intention(nom) for nom in _banking77("test").features["label"].names]


def banking77(split: str, n: int, graine: int = 0) -> list[dict]:
    """Message client -> intention (format commun, la réponse attendue est le nom de l'intention)."""
    ds = _banking77(split)
    noms = [intention(nom) for nom in ds.features["label"].names]
    ds = ds.shuffle(seed=graine).select(range(min(n, len(ds))))
    return [{"question": ex["text"], "reponse": noms[ex["label"]], "cible": noms[ex["label"]]} for ex in ds]


def _clinc150(split: str):
    """CLINC150 (150 intentions d'un assistant vocal) sans la classe hors périmètre « oos »."""
    from datasets import load_dataset

    ds = load_dataset("clinc/clinc_oos", "plus", split=split)
    oos = ds.features["intent"].names.index("oos")
    return ds.filter(lambda e: e["intent"] != oos)


def intentions_clinc150() -> list[str]:
    return [intention(n) for n in _clinc150("test").features["intent"].names if n != "oos"]


def clinc150(split: str, n: int, graine: int = 0) -> list[dict]:
    """Demande à un assistant -> intention parmi 150 (même format que banking77)."""
    ds = _clinc150(split)
    noms = [intention(n) for n in ds.features["intent"].names]
    ds = ds.shuffle(seed=graine).select(range(min(n, len(ds))))
    return [{"question": ex["text"], "reponse": noms[ex["intent"]], "cible": noms[ex["intent"]]} for ex in ds]


def entites_bio(mots: list[str], etiquettes: list[str]) -> list[tuple[str, str]]:
    """Entités (type, texte) d'une phrase étiquetée en BIO (B-PER, I-PER, O...)."""
    res, courant, type_ = [], [], None
    for mot, e in zip(mots, etiquettes):
        if e.startswith("B-") or (e.startswith("I-") and e[2:] != type_):
            if courant:
                res.append((type_, " ".join(courant)))
            courant, type_ = [mot], e[2:]
        elif e.startswith("I-"):
            courant.append(mot)
        else:
            if courant:
                res.append((type_, " ".join(courant)))
            courant, type_ = [], None
    if courant:
        res.append((type_, " ".join(courant)))
    return res


def liste_entites(entites: list[tuple[str, str]]) -> str:
    """Forme écrite d'une liste d'entités : une ligne « TYPE: texte » par entité, dans l'ordre de la phrase."""
    return "\n".join(f"{t}: {x}" for t, x in entites)


def lire_entites(sortie: str) -> set[tuple[str, str]]:
    """Entités lues dans une réponse (lignes « TYPE: texte »), sans tenir compte de la casse ni de l'ordre."""
    res = set()
    for ligne in sortie.splitlines():
        t, sep, x = ligne.partition(":")
        if sep and t.strip() and x.strip():
            res.add((t.strip().upper(), x.strip().lower()))
    return res


def conll2003(split: str, n: int, graine: int = 0) -> list[dict]:
    """Phrase -> entités nommées (PER, ORG, LOC, MISC) ; les phrases sans entité sont écartées."""
    from datasets import load_dataset

    ds = load_dataset("tomaarsen/conll2003", split="train" if split == "train" else "test")
    noms = ds.features["ner_tags"].feature.names
    res = []
    for ex in ds.shuffle(seed=graine):
        ents = entites_bio(ex["tokens"], [noms[i] for i in ex["ner_tags"]])
        if ents:
            res.append({"question": " ".join(ex["tokens"]), "reponse": liste_entites(ents),
                        "cible": "\n".join(sorted(f"{t}: {x.lower()}" for t, x in ents))})
        if len(res) >= n:
            break
    return res


def etiquettes(d) -> list[str] | None:
    """Liste fermée des réponses possibles d'une source de classification, sinon None."""
    return {"banking77": intentions_banking77, "clinc150": intentions_clinc150}.get(d.get("source"), lambda: None)()


def amorce_gsm8k(a) -> str | None:
    """Exemples few-shot pour un modèle de base : problèmes du jeu d'entraînement avec leur solution rédigée, au format
    « Question: ... Answer: ... ». `a` est la section `amorce` d'une config (n, graine) ; None si elle est absente."""
    if not a or not a.get("n"):
        return None
    exemples = gsm8k(a["n"], "train", graine=a.get("graine", 999))
    return "\n\n".join(f"Question: {e['question']}\nAnswer: {e['reponse']}" for e in exemples)


def consignes_generales(n: int, graine: int = 0, longueur_max: int = 300) -> list[str]:
    """Demandes générales variées (Alpaca, sans champ d'entrée) : prompts du remélange contre l'oubli."""
    from datasets import load_dataset

    ds = load_dataset("tatsu-lab/alpaca", split="train")
    ds = ds.filter(lambda e: not e["input"] and len(e["instruction"]) < longueur_max).shuffle(seed=graine)
    return ds.select(range(min(n, len(ds))))["instruction"]


def charger_qa(d, split: str = "train", n: int | None = None) -> list[dict]:
    """Point d'entrée unique : `d` est la section `donnees` de la config."""
    graine = d.get("graine", 0) + (0 if split == "train" else 10_000)
    n = n or d.get(f"n_{split}", 100)
    source = d.get("source", "arith")
    if source == "arith":
        return arith(n, graine, d.get("niveau", 2))
    if source == "gsm8k":
        return gsm8k(n, "train" if split == "train" else "test", graine)
    if source == "banking77":
        return banking77("train" if split == "train" else "test", n, graine)
    if source == "clinc150":
        return clinc150("train" if split == "train" else "test", n, graine)
    if source == "conll2003":
        return conll2003("train" if split == "train" else "test", n, graine)
    if source == "faits":
        return faits_fictifs(d.get("n_entites", 20), d.get("graine", 0))["qa"][:n]
    if source == "hf":
        return hf_qa(d["nom"], d.get(f"split_{split}", split), n, d["champs"], d.get("config"), graine, d.get("fichiers"))
    raise ValueError(f"source de données inconnue : {source}")


# --------------------------------------------------------------------------- préférences


def preferences_arith(n: int, graine: int = 0, niveau: int = 2) -> list[dict]:
    rng = random.Random(graine + 7)
    return [{"question": ex["question"], "chosen": ex["reponse"], "rejected": reponse_erronee(ex, rng), "cible": ex["cible"]}
            for ex in arith(n, graine, niveau)]


def preferences_hf(nom: str, split: str, n: int, graine: int = 0) -> list[dict]:
    """Datasets de préférences au format TRL (ex. trl-lib/ultrafeedback_binarized) : colonnes chosen/rejected."""
    from datasets import load_dataset

    ds = load_dataset(nom, split=split).shuffle(seed=graine).select(range(n))
    return [dict(ex) for ex in ds]
