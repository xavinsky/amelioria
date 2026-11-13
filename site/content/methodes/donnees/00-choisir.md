---
title: "DATA : choisir entre les méthodes"
nav: Choisir dans DATA
id: choisir-data
description: "Pré-entraînement continu, fine-tuning supervisé ou LoRA : comment choisir la méthode qui entraîne un modèle d'IA sur ses propres données."
comparatif: donnees
decision:
  q: "Que contiennent\nvos données ?"
  choix:
    "Des faits\nà consulter": {m: 18}
    "Du texte brut\ndu domaine":
      m: 1
      choix:
        puis la tâche: {m: 3}
    "Des exemples\nentrée → sortie":
      m: 3
      choix:
        "pas assez\nd'effet": {m: 2}
---

Les trois méthodes de cette catégorie font la même chose : elles **ajustent les poids du modèle sur vos données**, par descente de gradient. Ce qui les distingue, c'est la **nature des données** (texte brut ou exemples annotés) et la **part des poids modifiée** (tous, ou un petit adaptateur).

## Vous avez du texte brut : pré-entraînement continu

Des documents du domaine, sans questions ni réponses : rapports, contrats, articles médicaux, textes dans une langue mal couverte. Le [pré-entraînement continu](#pre-entrainement-continu) reprend l'entraînement du modèle sur ce texte pour qu'il en apprenne le vocabulaire et le style.

C'est la méthode la plus coûteuse de la catégorie : il faut beaucoup de texte (des dizaines de millions de tokens au moins) et plusieurs GPU pendant des jours. Elle rend le modèle plus à l'aise dans le domaine, mais **ne le rend pas capable de répondre à des questions sur ce texte** : l'exemple du dépôt le montre, la perplexité du domaine s'effondre sans que le modèle sache restituer les faits appris. On la fait donc suivre d'un entraînement sur la tâche visée.

## Vous avez des exemples entrée → sortie : LoRA d'abord

Des paires question → réponse, texte → catégorie, demande → JSON. On commence par [LoRA ou QLoRA](#peft) : seul un petit adaptateur est entraîné, sur un seul GPU, en quelques heures, et l'on peut garder un adaptateur par tâche ou par client.

Si l'effet ne suffit pas, notamment pour un gros changement de comportement, on passe au [fine-tuning supervisé complet](#fine-tuning-supervise), **sur les mêmes données** : tous les poids bougent, ce qui demande plus de mémoire et dégrade davantage les capacités générales.

## Avant de commencer

- **Vos données sont des faits qui changent ou qu'il faut citer** (catalogue, documentation interne) : ne les apprenez pas dans les poids, faites-les consulter par [RAG](#rag).
- **Essayez d'abord le prompt** ([EXEC17](#optimisation-prompt)) : s'il suffit, aucun entraînement n'est nécessaire.
- **Contre l'oubli** : le pré-entraînement continu et le fine-tuning complet dégradent ce que le modèle savait. On mélange à ses données une part de texte ou d'exemples généraux.
