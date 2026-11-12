---
title: "Cas : limites de la forme"
id: cas-forme
description: "Lever les limites de forme d'un modèle d'IA : textes trop longs, langue mal découpée, capacité insuffisante."
decision:
  q: "Quelle limite ?"
  choix:
    Textes trop longs:
      m: 18
      choix:
        tout le texte requis: {m: 12}
    Langue mal découpée:
      m: 13
      choix:
        puis: {m: 1}
    Capacité trop faible:
      m: 14
      choix:
        puis: {m: 1}
---
