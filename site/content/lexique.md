## Adaptateur

Petit module entraînable ajouté à un modèle gelé ; on l'entraîne, le déploie et le retire sans toucher aux poids d'origine. Voir [PEFT](#peft).

## BM25

Score de recherche lexicale qui pondère les mots de la requête par leur rareté dans le corpus. Brique de base du [RAG](#rag).

## Embedding

Vecteur qui représente le sens d'un texte ; deux textes proches ont des vecteurs proches. Sert à la recherche dense du [RAG](#rag).

## GGUF

Format de fichier de llama.cpp et d'Ollama pour les modèles quantifiés, exécutables sur CPU ou GPU. Voir [Quantification](#quantification).

## GPTQ

Quantification calibrée qui corrige les erreurs d'arrondi colonne par colonne à partir de quelques centaines de textes. Voir [Quantification](#quantification).

## LoRA

Low-Rank Adaptation : ajoute à chaque couche linéaire deux petites matrices de rang faible, seules entraînées. Voir [PEFT](#peft).

## NF4

Format 4 bits dont les niveaux suivent une distribution normale, celle des poids d'un réseau. Utilisé par QLoRA. Voir [Quantification](#quantification).

## QLoRA

LoRA entraînée sur un modèle chargé en 4 bits (NF4) : divise la mémoire par quatre environ. Voir [PEFT](#peft).

## Quantification

Stocker les poids (et parfois les activations) sur 8, 4 ou 3 bits au lieu de 16. Voir [Quantification](#quantification).

## RAG

Retrieval-Augmented Generation : chercher des passages pertinents et les placer dans le prompt au moment de répondre. Voir [RAG](#rag).

## Reranker

Modèle (cross-encoder) qui relit chaque couple question-passage pour reclasser les candidats d'une recherche. Voir [RAG](#rag).

## RRF

Reciprocal Rank Fusion : fusionne plusieurs classements en sommant l'inverse des rangs. Voir [RAG](#rag).

## SFT

Supervised Fine-Tuning : entraîner un modèle sur des paires instruction-réponse écrites à l'avance. La [LoRA](#peft) en est la version économe.
