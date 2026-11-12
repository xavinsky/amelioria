---
title: "Cas : sans réentraîner"
id: cas-sans-reentrainer
description: "Modifier un modèle d'IA sans le réentraîner : fusion de modèles, pilotage par activations."
decision:
  q: "Que dispose-t-on ?"
  choix:
    "Plusieurs fine-tunes\nd'une même base":
      m: 8
      choix:
        compétence perdue: {fin: "Entraîner sur\nl'ensemble des données"}
    "Quelques paires\ncontrastées":
      m: 11
      choix:
        effet instable: {m: 4}
---
