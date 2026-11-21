"""RAG (retrieval-augmented generation) : aller chercher les documents pertinents et les donner au modèle dans le prompt.

Recherche (`--set recherche.methode=...`) : bm25 (lexical), dense (embeddings), hybride (fusion RRF), + reranker optionnel.
Étapes : indexer le corpus -> mesurer sans RAG, avec RAG et avec des passages tirés au hasard (témoin) -> rappel de la
recherche, exactitude des réponses, confusions avec le personnage au nom voisin, refus sur les questions sans réponse.

Le corpus contient, pour chaque personnage interrogé, un « jumeau » : un personnage au nom presque identique (même
prénom, fin du nom changée) et aux faits tous différents. Une recherche approximative ramène ses passages, et le
modèle répond alors avec les faits du jumeau. Des questions portent aussi sur des personnages absents du corpus : la
bonne réponse est de dire qu'on ne sait pas.
"""

import collections
import math
import pathlib
import random
import re
import sys

sys.path.insert(0, str(next(p for p in pathlib.Path(__file__).resolve().parents if (p / "commun").is_dir())))

from commun import config, donnees, evaluation, preuve, rapport
from commun.generateur import Generateur


def mots(t):
    return re.findall(r"\w+", t.lower())


class BM25:
    """Okapi BM25 en quelques lignes : score lexical classique, très solide sur les noms propres et termes rares."""

    def __init__(self, docs, k1=1.5, b=0.75):
        self.docs = [mots(d) for d in docs]
        self.k1, self.b = k1, b
        self.moy = sum(map(len, self.docs)) / len(self.docs)
        df = collections.Counter(m for d in self.docs for m in set(d))
        self.idf = {m: math.log(1 + (len(self.docs) - n + 0.5) / (n + 0.5)) for m, n in df.items()}
        self.tf = [collections.Counter(d) for d in self.docs]

    def chercher(self, requete, k):
        q = mots(requete)
        scores = []
        for i, (tf, d) in enumerate(zip(self.tf, self.docs)):
            s = sum(self.idf.get(m, 0) * tf[m] * (self.k1 + 1) / (tf[m] + self.k1 * (1 - self.b + self.b * len(d) / self.moy)) for m in q if m in tf)
            scores.append((s, i))
        return [i for _, i in sorted(scores, reverse=True)[:k]]


class Dense:
    def __init__(self, docs, nom):
        from sentence_transformers import SentenceTransformer

        self.enc = SentenceTransformer(nom)
        self.emb = self.enc.encode(docs, normalize_embeddings=True, convert_to_tensor=True, batch_size=64, show_progress_bar=False)

    def chercher(self, requete, k):
        q = self.enc.encode([requete], normalize_embeddings=True, convert_to_tensor=True)
        return (q @ self.emb.T)[0].topk(min(k, self.emb.shape[0])).indices.tolist()


def fusion_rrf(listes, k, constante=60):
    """Reciprocal Rank Fusion : combine plusieurs classements sans calibrer leurs scores."""
    score = collections.defaultdict(float)
    for liste in listes:
        for rang, i in enumerate(liste):
            score[i] += 1 / (constante + rang + 1)
    return [i for i, _ in sorted(score.items(), key=lambda x: -x[1])[:k]]


REFUS = re.compile(r"don'?t know|do not know|not (?:mentioned|provided|specified|stated|available|in the (?:given )?context)"
                   r"|no information|cannot (?:answer|determine)|unknown", re.I)


def jumeau(e, rng, noms):
    """Personnage au nom presque identique (même prénom, fin du nom changée) dont tous les faits diffèrent."""
    prenom, famille = e["nom"].split(" ", 1)
    while True:
        nom = f"{prenom} {famille[:-2]}{rng.choice(['ka', 'vel', 'tor', 'mor', 'dra', 'fen', 'gol', 'rix', 'sol', 'bru', 'wen', 'lum'])}"
        if nom not in noms:
            break
    noms.add(nom)
    ville = e["ville"]
    while ville == e["ville"]:
        ville = donnees._nom_invente(rng).split()[0] + "ia"
    annee = e["annee"]
    while annee == e["annee"]:
        annee = str(rng.randint(1650, 1950))
    return {"nom": nom, "ville": ville, "annee": annee,
            "metier": rng.choice([m for m in donnees._METIERS if m != e["metier"]]),
            "creation": rng.choice([c for c in donnees._CREATIONS if c != e["creation"]])}


def main():
    cfg = config.charger(__file__)
    r = cfg.recherche
    d = cfg.donnees
    graine = d.graine + cfg.graine - 42   # 42 = graine par défaut : elle reproduit le corpus de la configuration
    kb = donnees.faits_fictifs(d.n_entites + d.n_sans_reponse, graine)
    presentes = kb["entites"][: d.n_entites]
    noms_presents = {e["nom"] for e in presentes}
    rng = random.Random(graine + 1)
    noms = {e["nom"] for e in kb["entites"]}
    jumeaux = {e["nom"]: jumeau(e, rng, noms) for e in presentes[: d.n_jumeaux]}
    # Corpus : une fiche par formulation, celles des jumeaux, + des paragraphes Wikipédia sans rapport.
    corpus = [{"texte": x["texte"], "entite": x["entite"]} for x in kb["documents"] if x["entite"] in noms_presents]
    for j in jumeaux.values():
        champs = dict(j, creation_maj=j["creation"][0].upper() + j["creation"][1:])
        corpus += [{"texte": g.format(**champs), "entite": j["nom"]} for g in donnees._GABARITS_DOC]
    corpus += [{"texte": t, "entite": None} for t in donnees.textes_generaux(d.n_distracteurs, graine=5)]
    textes = [c["texte"] for c in corpus]
    questions = [q for q in kb["qa"] if q["entite"] in noms_presents][: cfg.evaluation.n]
    # Questions sur des personnages absents du corpus : un seul fait demandé par personnage, la bonne réponse est un refus.
    sans_reponse = [[q for q in kb["qa"] if q["entite"] == e["nom"]][i % 4] for i, e in enumerate(kb["entites"][d.n_entites:])]
    print(f"[rag] corpus de {len(corpus)} passages, {len(questions)} questions, {len(sans_reponse)} sans réponse")

    bm25 = BM25(textes) if r.methode in ("bm25", "hybride") else None
    dense = Dense(textes, r.embeddings) if r.methode in ("dense", "hybride") else None
    reranker = None
    if r.get("reranker"):
        from sentence_transformers import CrossEncoder

        reranker = CrossEncoder(r.reranker)

    def chercher(q):
        candidats = r.k_candidats if reranker else r.k
        if r.methode == "bm25":
            ids = bm25.chercher(q, candidats)
        elif r.methode == "dense":
            ids = dense.chercher(q, candidats)
        else:
            ids = fusion_rrf([bm25.chercher(q, candidats), dense.chercher(q, candidats)], candidats)
        if reranker:
            scores = reranker.predict([(q, textes[i]) for i in ids])
            ids = [i for _, i in sorted(zip(scores, ids), reverse=True)]
        return ids[: r.k]

    trouves = [chercher(q["question"]) for q in questions]
    rappel = sum(any(corpus[i]["entite"] == q["entite"] for i in ids) for q, ids in zip(questions, trouves)) / len(questions)
    trouves_sr = [chercher(q["question"]) for q in sans_reponse]
    hasard = random.Random(graine + 2)
    tires = [hasard.sample(range(len(textes)), r.k) for _ in questions + sans_reponse]

    gen = Generateur.depuis_config(cfg.modele)
    ev = cfg.evaluation

    def demander(qs, passages):
        """Réponses du modèle ; passages=None : sans contexte (témoin sans RAG)."""
        convs = [[{"role": "user", "content": q["question"] + " Answer briefly." if ids is None else cfg.gabarit.format(
            contexte="\n".join(f"[{j + 1}] {textes[i]}" for j, i in enumerate(ids)), question=q["question"])}]
            for q, ids in zip(qs, passages or [None] * len(qs))]
        return gen.chat(convs, max_new_tokens=ev.max_new_tokens)

    def exactitude(sorties):
        ok = [evaluation.correct(s, q["cible"], "contient") for s, q in zip(sorties, questions)]
        return {**preuve.taux(ok), "exemple": sorties[0].strip()[:200]}

    s_sans, s_avec = demander(questions, None), demander(questions, trouves)
    s_hasard = demander(questions, tires[: len(questions)])
    sans, avec, alea = exactitude(s_sans), exactitude(s_avec), exactitude(s_hasard)

    # Confusion avec le jumeau : la réponse donne le fait du personnage au nom voisin et non celui du personnage interrogé.
    confondus = [bool(q["entite"] in jumeaux and str(jumeaux[q["entite"]][q["attribut"]]).lower() in s.lower()
                      and str(q["cible"]).lower() not in s.lower()) for q, s in zip(questions, s_avec)]
    n_avec_jumeau = sum(q["entite"] in jumeaux for q in questions)

    # Questions sans réponse : part de refus (« je ne sais pas ») sans RAG, avec RAG et avec passages tirés au hasard.
    refus = {nom: [bool(REFUS.search(s)) for s in sorties] for nom, sorties in
             (("sans_rag", demander(sans_reponse, None)), ("avec_rag", demander(sans_reponse, trouves_sr)),
              ("passages_aleatoires", demander(sans_reponse, tires[len(questions):])))}
    sans_rep = {nom: preuve.taux(v, "refus") for nom, v in refus.items()}
    sans_rep["avec_rag_moins_sans_rag"] = preuve.difference_appariee(refus["sans_rag"], refus["avec_rag"])

    rapport.enregistrer(cfg, sans_rag=sans, avec_rag=avec | {f"rappel_a_{r.k}": round(rappel, 4),
                                                              "confondus_avec_le_jumeau": round(sum(confondus) / max(n_avec_jumeau, 1), 4)},
                        passages_aleatoires=alea, sans_reponse=sans_rep,
                        recherche=f"{r.methode}{' + reranker' if reranker else ''}")


if __name__ == "__main__":
    main()
