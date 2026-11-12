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

Trois limites tiennent à la forme du modèle plutôt qu'à ce qu'il sait.

## Textes trop longs

Le [RAG](#rag) contourne la fenêtre de contexte en ne fournissant que les passages utiles. Quand il faut vraiment **tout le texte** à la fois (contrat complet, base de code, long dialogue), l'**extension de contexte** apprend au modèle à lire des textes plus longs que ceux de son entraînement, par un court entraînement sur des textes longs.

## Langue mal découpée

Un modèle lit le texte par morceaux de mots (tokens). Une langue ou un jargon qu'il découpe en trop de petits morceaux coûte plus cher et se comprend moins bien. L'**extension de vocabulaire** lui ajoute des morceaux adaptés, **puis** un pré-entraînement continu apprend au modèle à s'en servir.

## Capacité trop faible

Quand le modèle sature mais qu'on ne veut pas repartir d'un pré-entraînement complet, la **croissance d'architecture** agrandit le modèle en copiant certaines de ses couches, **puis** un pré-entraînement continu apprend au modèle agrandi à s'en servir.
