"""Indexation d'un dossier de documents (txt, md) dans Qdrant : découpage en passages, embeddings via TEI.

    pip install qdrant-client requests
    python 18-rag/infra/indexer_qdrant.py --dossier mes_documents/ --collection docs
"""

import argparse
import pathlib
import uuid

import requests


def decouper(texte, taille=800, chevauchement=150):
    """Passages de ~taille caractères avec chevauchement, coupés sur des fins de paragraphe quand c'est possible."""
    morceaux, debut = [], 0
    while debut < len(texte):
        fin = min(debut + taille, len(texte))
        coupe = texte.rfind("\n\n", debut, fin)
        if coupe > debut + taille // 2:
            fin = coupe
        morceaux.append(texte[debut:fin].strip())
        debut = max(fin - chevauchement, debut + 1)
    return [m for m in morceaux if m]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dossier", required=True)
    p.add_argument("--collection", default="docs")
    p.add_argument("--tei", default="http://localhost:8081")
    p.add_argument("--qdrant", default="http://localhost:6333")
    a = p.parse_args()

    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, PointStruct, VectorParams

    client = QdrantClient(url=a.qdrant)
    passages = []
    for f in pathlib.Path(a.dossier).rglob("*"):
        if f.suffix in {".txt", ".md"}:
            passages += [{"source": str(f), "texte": m} for m in decouper(f.read_text(encoding="utf-8", errors="ignore"))]
    print(f"{len(passages)} passages")

    dim = None
    for i in range(0, len(passages), 64):
        lot = passages[i : i + 64]
        vecteurs = requests.post(f"{a.tei}/embed", json={"inputs": [x["texte"] for x in lot], "normalize": True}).json()
        if dim is None:
            dim = len(vecteurs[0])
            if not client.collection_exists(a.collection):
                client.create_collection(a.collection, vectors_config=VectorParams(size=dim, distance=Distance.COSINE))
        client.upsert(a.collection, [PointStruct(id=str(uuid.uuid4()), vector=v, payload=x) for v, x in zip(vecteurs, lot)])


if __name__ == "__main__":
    main()
