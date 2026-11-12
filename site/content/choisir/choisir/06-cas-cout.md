---
title: "Cas : trop gros ou trop lent"
id: cas-cout
description: "Réduire la taille ou le coût d'un modèle d'IA : quantification, élagage, distillation."
decision:
  q: "Que faut-il réduire ?"
  choix:
    La mémoire:
      m: 15
      choix:
        perte trop forte: {fin: "8 bits, ou méthode\ncalibrée (GPTQ, AWQ)"}
    La taille du modèle:
      q: "Un modèle plus petit\nde la même famille ?"
      choix:
        Oui: {m: 6}
        Non:
          m: 16
          choix:
            puis: {m: 6}
---

Deux façons de rendre un modèle moins cher : le même modèle stocké de façon plus compacte, ou un modèle réellement plus petit.

## Réduire la mémoire : la quantification

La [quantification](#quantification) stocke les poids sur 8 ou 4 bits au lieu de 16. En quelques minutes et sans données, la mémoire est divisée par 2 à 4, souvent avec un débit meilleur. Si la **perte de qualité est trop forte** en 4 bits, on repasse en 8 bits ou on utilise une méthode calibrée (GPTQ, AWQ), qui protège les poids les plus sensibles.

## Réduire la taille du modèle

S'il existe un **modèle plus petit de la même famille**, la **distillation** lui apprend à imiter le grand : l'élève hérite d'une bonne partie de la qualité de l'enseignant pour une fraction du coût. Sinon, l'**élagage** retire du grand modèle des couches ou des neurones peu utiles, **puis** une distillation depuis le modèle d'origine récupère la qualité perdue.
