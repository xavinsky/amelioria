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
