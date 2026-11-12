---
title: RAG
id: rag
dossier: 18-rag
aka: Retrieval-Augmented Generation
schema:
  size: [770, 250]
  nodes:
    docs: {label: Documents, sub: "fiches, wiki, PDF", external: true, x: 20, y: 24, w: 160}
    index: {label: Index, sub: "BM25 + embeddings", x: 20, y: 168, w: 160}
    question: {label: Question, external: true, x: 226, y: 24, w: 140}
    recherche: {label: Recherche hybride, sub: "fusion RRF, top 20", focus: true, x: 226, y: 168, w: 168}
    rerank: {label: Reranker, sub: "cross-encoder, top 3", x: 436, y: 168, w: 150}
    llm: {label: LLM gelé, sub: "prompt + 3 passages", x: 612, y: 96, w: 146}
  flows:
    - {from: docs, to: index, text: "Les documents sont découpés en passages et indexés deux fois : par leurs mots (BM25) et par leur sens (embeddings)."}
    - {from: question, to: recherche, text: "La question interroge les deux index ; leurs classements sont fusionnés par rang (RRF)."}
    - {from: index, to: recherche, text: "Une vingtaine de passages candidats remontent."}
    - {from: recherche, to: rerank, text: "Un cross-encoder relit chaque couple question-passage et garde les trois meilleurs."}
    - {from: rerank, to: llm, text: "Le prompt contient la consigne, les trois passages et la question ; le modèle répond à partir de ces sources."}
references:
  "Lewis et al., Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (2020)": https://arxiv.org/abs/2005.11401
  "Robertson & Zaragoza, The Probabilistic Relevance Framework: BM25 and Beyond (2009)": https://doi.org/10.1561/1500000019
  "Karpukhin et al., Dense Passage Retrieval (2020)": https://arxiv.org/abs/2004.04906
  "Chen et al., BGE-M3 (2024)": https://arxiv.org/abs/2402.03216
  "Cormack, Clarke & Buettcher, Reciprocal Rank Fusion (2009)": https://doi.org/10.1145/1571941.1572114
  "Nogueira & Cho, Passage Re-ranking with BERT (2019)": https://arxiv.org/abs/1901.04085
  "Liu et al., Lost in the Middle: How Language Models Use Long Contexts (2023)": https://arxiv.org/abs/2307.03172
  "Gao et al., Retrieval-Augmented Generation for Large Language Models: A Survey (2023)": https://arxiv.org/abs/2312.10997
---

Au moment de répondre, on cherche les passages pertinents dans une base documentaire et on les place dans le prompt. Le modèle n'est pas modifié : il répond à partir de sources à jour, privées et citables. C'est la méthode standard pour brancher des données d'entreprise sur un modèle, y compris un modèle fermé appelé par API.

## Principe

Tout se joue dans la recherche : si le bon passage n'est pas dans le prompt, aucun modèle ne répondra bien. Une recherche solide combine plusieurs briques.

| Brique | Rôle |
|---|---|
| **BM25** | Correspondance de mots, pondérée par leur rareté. Excellent sur les noms propres, les codes, les termes exacts |
| **Recherche dense** | Similarité de sens entre embeddings. Trouve les paraphrases, les synonymes, les autres langues |
| **Fusion RRF** | Combine les deux classements par leurs rangs, sans avoir à calibrer des scores de nature différente |
| **Reranker** | Un cross-encoder lit chaque couple question-passage : plus précis mais plus lent, il n'est appliqué qu'aux candidats |

On mesure séparément la recherche (**rappel@k** : le bon passage est-il parmi les k fournis ?) et la réponse finale (exactitude, fidélité aux sources).

## Quand l'utiliser

- Connaissances **privées, volumineuses ou qui changent** : documentation interne, catalogue produit, juridique, support.
- Besoin de **citer ses sources** et de limiter les inventions du modèle.
- Mise à jour immédiate : on réindexe, sans réentraîner.
- Textes plus longs que la fenêtre de contexte : on ne fournit que les passages utiles.

**À éviter** pour changer un comportement, un style ou une compétence de raisonnement : c'est le rôle de l'entraînement, par exemple une [LoRA](#peft). Pour quelques faits figés à corriger dans le modèle lui-même, l'édition de connaissances est plus directe.

## Pièges

- **Découpage** : des passages trop petits perdent le contexte ; trop grands, ils diluent l'information et coûtent des tokens. 300 à 1 000 caractères avec chevauchement est un bon départ.
- **Embeddings anglophones sur du français** : prendre un modèle multilingue (bge-m3, multilingual-e5).
- **Contexte ignoré** : le modèle préfère parfois ses propres connaissances. Le dire explicitement dans la consigne, avec la permission de répondre « je ne sais pas ».
- **Perdu au milieu** : les passages placés au milieu d'un long contexte sont moins utilisés. Garder peu de passages, bien classés.
