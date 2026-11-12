---
title: "Cas : raisonnement"
id: cas-raisonnement
description: "Améliorer le raisonnement d'un modèle d'IA (maths, code) : outils, calcul à l'inférence, données synthétiques, RL."
decision:
  q: "Calcul exact ou\naction nécessaire ?"
  choix:
    Oui: {m: 19}
    Non:
      m: 20
      choix:
        suffit: {fin: "Garder le vote,\npayer N appels"}
        trop cher à l'usage:
          q: "Réponses vérifiables\npar un programme ?"
          choix:
            Oui:
              m: 7
              choix:
                ne suffit pas: {m: 5}
            Non: {m: 2}
---
