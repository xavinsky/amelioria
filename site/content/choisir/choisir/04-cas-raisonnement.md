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

On distingue d'abord ce qui relève du calcul ou de l'action, que le modèle ne doit pas faire de tête, de ce qui relève vraiment du raisonnement.

## Calcul exact ou action : les outils

Un calcul exact, une requête en base, un appel d'API : le modèle décrit l'appel, un programme l'exécute et lui rend le résultat. C'est fiable, peu coûteux, et compatible avec un modèle fermé.

## Raisonnement : d'abord à l'inférence

Faire produire plusieurs raisonnements et garder la réponse majoritaire améliore nettement la justesse sans rien entraîner. Le prix se paie à chaque requête : N appels au lieu d'un. Si ça **suffit**, on garde le vote.

## Trop cher à l'usage : apprendre le raisonnement

Si le coût par requête devient trop élevé, on fait entrer le raisonnement dans les poids. Tout dépend alors de la possibilité de **vérifier une réponse par un programme** (résultat numérique, tests de code, requête SQL).

- **Réponses vérifiables** : on fait générer de nombreuses solutions, on ne garde que les justes et on entraîne dessus (**données synthétiques**). Si ce n'est **pas suffisant**, le **RL** à récompense vérifiable fait progresser le modèle au-delà de ce qu'il réussit déjà.
- **Réponses non vérifiables** : il faut des solutions rédigées par des humains, et un **SFT** sur ces exemples.
