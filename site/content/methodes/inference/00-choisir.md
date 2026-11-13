---
title: "EXEC : choisir entre les méthodes"
nav: Choisir dans EXEC
id: choisir-exec
description: "Optimisation du prompt, RAG, outils et agents ou calcul à l'inférence : comment améliorer un modèle d'IA, même fermé, sans toucher à ses poids."
comparatif: inference
decision:
  q: "Que manque-t-il\nau modèle ?"
  choix:
    "Des consignes\nclaires": {m: 17}
    "Des documents\nà consulter": {m: 18}
    "Agir ou calculer\nexactement": {m: 19}
    "De la fiabilité\nen raisonnement": {m: 20}
---

Ces méthodes ne modifient pas le modèle : elles améliorent **ce qui se passe autour de lui, au moment où il répond**. Elles marchent donc aussi avec un modèle fermé appelé par API, se mettent en place en quelques minutes et se défont aussi vite. Elles se combinent entre elles et avec toutes les autres catégories. **On commence toujours par là**, avant d'envisager un entraînement.

## Des consignes claires : optimisation du prompt

Le modèle sait faire la tâche mais la comprend mal : format non respecté, raisonnement bâclé, ton inadapté. L'[optimisation du prompt](#optimisation-prompt) améliore la consigne : instruction précise, raisonnement pas à pas, exemples choisis parmi les plus proches de la question. C'est gratuit et immédiat, mais cela ne donne pas au modèle une compétence qu'il n'a pas. Si le prompt doit contenir des dizaines d'exemples, un entraînement LoRA ([DATA03](#peft)) devient rentable.

## Des documents à consulter : RAG

Le modèle doit répondre sur des informations qu'il n'a jamais vues : documentation interne, catalogue, textes juridiques. Le [RAG](#rag) recherche les passages utiles et les place dans le prompt. Les connaissances se mettent à jour en réindexant, et les réponses peuvent citer leurs sources. La qualité dépend d'abord de la recherche : si le bon passage n'est pas retrouvé, le modèle ne peut pas bien répondre.

## Agir ou calculer exactement : outils et agents

Le modèle doit faire un calcul exact, lire une donnée à jour ou agir sur un système (requête SQL, envoi, création). Avec les [outils](#outils-agents), il décide d'appeler une fonction, lit le résultat et poursuit. Chaque appel relance le modèle, donc la latence et le coût augmentent. Un outil qui agit sur un système doit être isolé.

## De la fiabilité en raisonnement : calcul à l'inférence

Le modèle trouve souvent la bonne réponse, mais pas à chaque fois. Le [calcul à l'inférence](#calcul-inference) génère plusieurs réponses et garde la plus fréquente (vote majoritaire) ou la mieux notée. Le coût est multiplié par le nombre de réponses générées, et il faut des réponses comparables (un nombre, un choix). L'écart entre une réponse unique et la meilleure de N indique aussi ce qu'un entraînement par RL ([SIGN05](#rl-en-ligne)) pourrait apporter.
