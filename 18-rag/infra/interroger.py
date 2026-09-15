"""Question -> recherche Qdrant -> reranking TEI -> réponse vLLM avec citations.

    python 18-rag/infra/interroger.py "Quelle est la politique de congés ?"
"""

import argparse

import requests


def main():
    p = argparse.ArgumentParser()
    p.add_argument("question")
    p.add_argument("--collection", default="docs")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--modele", default="Qwen/Qwen2.5-72B-Instruct-AWQ")
    a = p.parse_args()

    from qdrant_client import QdrantClient

    q_vec = requests.post("http://localhost:8081/embed", json={"inputs": [a.question], "normalize": True}).json()[0]
    candidats = QdrantClient(url="http://localhost:6333").query_points(a.collection, query=q_vec, limit=40).points
    textes = [c.payload["texte"] for c in candidats]
    classement = requests.post("http://localhost:8082/rerank", json={"query": a.question, "texts": textes}).json()
    meilleurs = [candidats[r["index"]] for r in sorted(classement, key=lambda r: -r["score"])[: a.k]]

    contexte = "\n\n".join(f"[{i + 1}] ({c.payload['source']}) {c.payload['texte']}" for i, c in enumerate(meilleurs))
    consigne = (f"Réponds uniquement à partir du contexte, en citant les sources [n]. Si le contexte ne suffit pas, dis-le.\n\n"
                f"Contexte :\n{contexte}\n\nQuestion : {a.question}")
    rep = requests.post("http://localhost:8000/v1/chat/completions",
                        json={"model": a.modele, "messages": [{"role": "user", "content": consigne}], "temperature": 0}).json()
    print(rep["choices"][0]["message"]["content"])


if __name__ == "__main__":
    main()
