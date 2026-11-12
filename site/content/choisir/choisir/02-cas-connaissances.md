---
title: "Cas : ses connaissances"
id: cas-connaissances
description: "Ajouter, corriger ou retirer des connaissances à un modèle d'IA : RAG, pré-entraînement continu, SFT, édition, désapprentissage."
decision:
  q: "Que faut-il changer\ndans ses connaissances ?"
  choix:
    Ajouter:
      m: 18
      choix:
        suffit: {fin: "Garder le RAG,\nréindexer au besoin"}
        ne suffit pas:
          m: 1
          choix:
            puis: {m: 2}
    Corriger:
      q: "Quelques faits\nprécis ?"
      choix:
        Oui: {m: 9}
        Non, beaucoup: {m: 3}
    Retirer:
      m: 10
      choix:
        oubli incomplet: {fin: "Filtrer en sortie, ou\nréentraîner sans"}
---

La première question est la nature du changement : apporter un savoir que le modèle n'a pas, rectifier ce qu'il croit savoir, ou lui faire oublier quelque chose.

## Ajouter des connaissances

On commence toujours par le [RAG](#rag) : les documents restent hors du modèle, on les cherche au moment de répondre et on les place dans le prompt. Rien n'est entraîné, la mise à jour se fait en réindexant, et la réponse peut citer ses sources. C'est le bon choix pour des données privées, volumineuses ou qui changent.

Le RAG **ne suffit pas** quand la recherche ne trouve pas le bon passage, quand il faut raisonner sur l'ensemble du corpus plutôt que sur trois extraits, ou quand le coût d'une recherche à chaque question est trop élevé. On fait alors entrer le savoir dans les poids par un **pré-entraînement continu** sur le corpus, mélangé à du texte général pour limiter l'oubli. Ce pré-entraînement donne au modèle les connaissances mais pas l'habitude de s'en servir pour répondre : il est **suivi d'un SFT** sur des questions-réponses portant sur ce savoir.

## Corriger des connaissances

Pour **quelques faits précis** (un nom, une date, un rattachement), l'**édition de connaissances** modifie directement l'endroit du modèle qui stocke le fait, en quelques secondes, sans toucher au reste. Elle ne tient pas l'échelle : au-delà de quelques dizaines de faits, les éditions se gênent. Pour **beaucoup de corrections**, on entraîne une [LoRA](#peft) sur des exemples corrigés.

## Retirer des connaissances

Le **désapprentissage** pousse le modèle à ne plus produire un savoir (donnée personnelle, contenu protégé) tout en préservant le reste, mesuré sur un jeu à garder. L'**oubli reste souvent incomplet** : le savoir peut ressortir sous une autre formulation. On ajoute alors un filtre en entrée et en sortie, et seul un réentraînement sans les données concernées garantit leur absence.
