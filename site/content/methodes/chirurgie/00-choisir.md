---
title: "EDIT : choisir entre les méthodes"
nav: Choisir dans EDIT
id: choisir-edit
description: "Fusion de modèles, édition de connaissances, désapprentissage ou pilotage (steering) par activations : comment choisir une intervention ciblée sur un modèle d'IA."
comparatif: chirurgie
decision:
  q: "Que faut-il\nchanger ?"
  choix:
    "Réunir des\nmodèles affinés": {m: 8}
    "Corriger\nquelques faits":
      m: 9
      choix:
        "faits nombreux\nou changeants": {m: 18}
    "Faire oublier\nun contenu": {m: 10}
    "Orienter un\ntrait (ton)":
      m: 11
      choix:
        instable: {m: 4}
---

Ces méthodes interviennent **directement sur un modèle existant**, de façon ciblée, sans long entraînement : quelques minutes de calcul, peu ou pas de données. Chacune répond à un besoin précis ; le choix se fait sur **ce qu'il faut changer**.

## Réunir plusieurs modèles : fusion

Plusieurs versions affinées **du même modèle de base** (une pour le code, une pour le support, une pour le juridique) et un seul modèle à servir. La [fusion](#fusion-modeles) combine leurs poids en quelques minutes sur CPU, sans données. Elle sert aussi à récupérer des capacités générales perdues pendant un fine-tuning, en refusionnant avec le modèle d'origine. Une fusion peut dégrader une compétence sans prévenir : il faut toujours mesurer le résultat.

## Corriger quelques faits : édition de connaissances

Un fait faux ou périmé (« le PDG de l'entreprise est X »). L'[édition de connaissances](#edition-connaissances) repère la couche où il est stocké et la modifie, sans toucher au reste. La correction reste locale : le modèle ne l'utilise pas toujours dans un raisonnement en plusieurs étapes. Pour des faits nombreux ou qui changent souvent, mieux vaut les faire consulter par [RAG](#rag).

## Faire oublier un contenu : désapprentissage

Une donnée personnelle à retirer, un contenu sous licence, une connaissance dangereuse. Le [désapprentissage](#desapprentissage) pousse le modèle à ne plus produire ce contenu tout en gardant le reste. Il n'offre **aucune garantie formelle** : l'information peut revenir par extraction ou par un léger fine-tuning. Si l'objectif est seulement de ne pas *dire* quelque chose, un filtre en sortie peut suffire.

## Orienter un trait : pilotage par activations

Le modèle fait ce qu'on attend, mais on veut le rendre plus formel, plus positif, plus concis. Le [pilotage par activations](#pilotage-activations) calcule une direction à partir de quelques paires de phrases et l'ajoute pendant la génération, avec une intensité réglable. Il réoriente ce que le modèle sait déjà, sans rien lui apprendre. Si l'effet est instable, on inscrit le trait dans les poids par [DPO](#optimisation-preferences), avec les mêmes paires.
