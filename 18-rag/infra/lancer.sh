#!/usr/bin/env bash
# RAG de production : services GPU (Qdrant, TEI embeddings + reranker, vLLM), indexation, interrogation, évaluation.
set -euo pipefail
cd "$(dirname "$0")/../.."
docker compose -f 18-rag/infra/docker-compose.yml up -d
until curl -s localhost:8000/v1/models > /dev/null; do sleep 10; done
python 18-rag/run.py --config 18-rag/infra/config.yaml
# Documents réels :
#   python 18-rag/infra/indexer_qdrant.py --dossier mes_documents/
#   python 18-rag/infra/interroger.py "ma question"
# Évaluation RAG dédiée (fidélité, pertinence du contexte) : ragas, ou un jeu question/réponse annoté + LLM juge.
