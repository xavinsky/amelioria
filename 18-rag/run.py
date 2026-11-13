"""RAG (retrieval-augmented generation) : aller chercher les documents pertinents et les donner au modèle dans le prompt.

Recherche (`--set recherche.methode=...`) : bm25 (lexical), dense (embeddings), hybride (fusion RRF), + reranker optionnel.
Étapes : indexer le corpus -> mesurer sans RAG -> mesurer avec RAG (rappel de la recherche + exactitude des réponses).
"""

import collections
import math
import pathlib
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


def main():
    cfg = config.charger(__file__)
    r = cfg.recherche
    kb = donnees.faits_fictifs(cfg.donnees.n_entites, cfg.donnees.graine)
    # Corpus : une fiche par formulation + des paragraphes Wikipédia sans rapport (distracteurs réalistes).
    corpus = [{"texte": d["texte"], "entite": d["entite"]} for d in kb["documents"]]
    corpus += [{"texte": t, "entite": None} for t in donnees.textes_generaux(cfg.donnees.n_distracteurs, graine=5)]
    textes = [c["texte"] for c in corpus]
    questions = kb["qa"][: cfg.evaluation.n]
    print(f"[rag] corpus de {len(corpus)} passages, {len(questions)} questions")

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

    gen = Generateur.depuis_config(cfg.modele)
    ev = cfg.evaluation

    def exactitude(convs):
        sorties = gen.chat(convs, max_new_tokens=ev.max_new_tokens)
        ok = [evaluation.correct(s, q["cible"], "contient") for s, q in zip(sorties, questions)]
        return {**preuve.taux(ok), "exemple": sorties[0].strip()[:200]}

    sans = exactitude([[{"role": "user", "content": q["question"] + " Answer briefly."}] for q in questions])
    avec = exactitude([[{"role": "user", "content": cfg.gabarit.format(
        contexte="\n".join(f"[{j + 1}] {textes[i]}" for j, i in enumerate(ids)), question=q["question"])}]
        for q, ids in zip(questions, trouves)])

    rapport.enregistrer(cfg, sans_rag=sans, avec_rag=avec | {f"rappel_a_{r.k}": round(rappel, 4)},
                        recherche=f"{r.methode}{' + reranker' if reranker else ''}")


if __name__ == "__main__":
    main()
