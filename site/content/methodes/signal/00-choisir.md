---
title: "SIGN : choisir entre les méthodes"
nav: Choisir dans SIGN
id: choisir-sign
description: "DPO, RL en ligne, distillation ou données synthétiques : comment choisir selon le signal dont on dispose pour améliorer un modèle d'IA."
comparatif: signal
decision:
  q: "Quel signal\navez-vous ?"
  choix:
    "Des paires\nA meilleur que B": {m: 4}
    "Un vérificateur\nautomatique":
      m: 7
      choix:
        plafonne: {m: 5}
    "Une note\n(humain ou juge)": {m: 5}
    "Un grand modèle\nenseignant": {m: 6}
---

Ici, on n'a pas les bonnes réponses écrites à l'avance. On a **un signal** qui dit ce qui est mieux : une préférence entre deux réponses, un programme qui vérifie le résultat, une note, ou un modèle plus fort qui montre l'exemple. **Le signal disponible choisit la méthode.**

## Des préférences : optimisation des préférences (DPO)

Pour une même demande, une réponse préférée et une réponse rejetée : choix d'annotateurs, 👍/👎 des utilisateurs, avis d'un modèle juge. L'[optimisation des préférences](#optimisation-preferences) apprend directement de ces paires, sans modèle de récompense ni génération pendant l'entraînement. C'est la méthode la plus simple et la plus stable pour régler un ton, une concision, une prudence. Elle s'applique en général après un fine-tuning supervisé.

## Un vérificateur : données synthétiques, puis RL

La réponse se vérifie automatiquement : un résultat de calcul, un code qui passe ses tests, un JSON valide.

- On commence par les [données synthétiques](#donnees-synthetiques) : le modèle génère plusieurs réponses, le vérificateur garde les justes, et l'on entraîne le modèle sur ce qu'il a réussi. C'est simple et peu coûteux.
- Quand ce filtrage plafonne, on passe au [RL en ligne](#rl-en-ligne) (GRPO, DAPO) : le modèle est récompensé à chaque essai réussi et améliore sa façon de chercher. C'est le plus efficace pour le raisonnement, mais aussi le plus coûteux.

Dans les deux cas, il faut que le modèle réussisse déjà de temps en temps. S'il échoue toujours, il n'y a rien à renforcer : il faut d'abord un fine-tuning supervisé ou une tâche plus facile.

## Une note : RL avec modèle de récompense

Sans vérificateur, mais avec des notes données par des humains ou par un modèle juge, on entraîne un modèle de récompense qui note les réponses, puis on optimise le modèle par [RL en ligne](#rl-en-ligne). Le risque : le modèle apprend à **exploiter les failles du juge** plutôt qu'à mieux répondre. Il faut surveiller ses sorties.

## Un modèle enseignant : distillation

Un grand modèle fait bien la tâche et l'on veut un petit modèle, moins cher à servir, qui s'en approche. La [distillation](#distillation) entraîne le petit modèle à imiter les sorties, ou les probabilités, du grand. L'élève reste en dessous de son enseignant.
