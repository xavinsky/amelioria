---
title: "FORM : choisir entre les méthodes"
nav: Choisir dans FORM
id: choisir-form
description: "Étendre le contexte ou le vocabulaire, agrandir, quantifier ou élaguer : comment choisir la méthode qui change la forme d'un modèle d'IA."
comparatif: remodeler
decision:
  q: "Dans quel sens ?"
  choix:
    Étendre:
      q: "Quelle limite\ngêne ?"
      choix:
        "Textes trop\nlongs": {m: 12}
        "Langue mal\ndécoupée": {m: 13}
        "Capacité\ninsuffisante": {m: 14}
    Compresser:
      m: 15
      choix:
        "encore trop\ngros ou lent": {m: 16}
---

Ces méthodes changent **la forme du modèle** : la longueur de texte qu'il lit, son vocabulaire, son nombre de couches, la précision de ses poids, sa taille. Elles vont dans deux sens opposés : **étendre** ce qu'il peut faire, ou **compresser** ce qu'il coûte.

## Étendre : trouver la limite qui gêne

- **Les textes sont plus longs que sa fenêtre** (contrats, code, longues conversations) : l'[extension de contexte](#extension-contexte) modifie l'encodage des positions, puis un court entraînement sur des textes longs apprend au modèle à s'en servir. Si seule une petite partie des documents est utile à chaque question, le [RAG](#rag) coûte bien moins cher.
- **La langue ou le domaine est mal découpé en tokens** (plus de 2 à 3 tokens par mot) : l'[extension de vocabulaire](#extension-vocabulaire) ajoute des tokens, puis un entraînement conséquent leur donne un sens. Pour quelques mots rares, c'est inutile.
- **Le modèle manque de capacité** : la [croissance d'architecture](#croissance-architecture) ajoute des couches ou des experts à un modèle existant. C'est la plus coûteuse : sans un long entraînement de cicatrisation, le modèle agrandi n'est pas meilleur que l'original.

Les trois demandent ensuite un entraînement sur du texte brut, avec une part de texte général contre l'oubli.

## Compresser : quantifier d'abord

- La [quantification](#quantification) stocke les poids sur 8 ou 4 bits au lieu de 16. Elle est rapide, sans entraînement, et la perte est faible en 8 bits, modérée en 4 bits sur les modèles de 7B et plus. C'est le premier réflexe pour faire tenir un modèle sur un matériel plus petit.
- Si le modèle reste trop gros ou trop lent, l'[élagage](#elagage) retire des poids, des couches ou des neurones, puis une cicatrisation, idéalement par distillation, restaure la qualité. Il ne fait gagner de la mémoire ou du temps que s'il est structuré ou servi par un moteur adapté. Les deux se combinent : élaguer, puis quantifier.
