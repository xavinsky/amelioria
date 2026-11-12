---
title: "Cas : trop gros ou trop lent"
id: cas-cout
description: "Réduire la taille ou le coût d'un modèle d'IA : quantification, élagage, distillation."
decision:
  q: "Réduire la mémoire ou\nle nombre de paramètres ?"
  choix:
    Mémoire:
      m: 15
      choix:
        perte trop forte: {fin: "8 bits, ou méthode\ncalibrée (GPTQ, AWQ)"}
    Paramètres:
      q: "Un modèle plus petit\nde la même famille ?"
      choix:
        Oui: {m: 6}
        Non:
          m: 16
          choix:
            puis: {m: 6}
---
