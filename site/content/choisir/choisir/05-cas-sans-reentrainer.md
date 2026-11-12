---
title: "Cas : sans réentraîner"
id: cas-sans-reentrainer
description: "Modifier un modèle d'IA sans le réentraîner : fusionner plusieurs modèles spécialisés en un seul, ou orienter son ton par pilotage des activations."
decision:
  q: "Que veut-on obtenir ?"
  choix:
    "Réunir plusieurs\nmodèles":
      q: "Issus du même\nmodèle de base ?"
      choix:
        Oui:
          m: 8
          choix:
            une compétence se perd: {fin: "Entraîner un modèle\nsur toutes les données"}
        Non: {fin: "Fusion impossible :\nservir plusieurs modèles"}
    "Changer un trait\n(ton, style)":
      m: 11
      choix:
        instable: {m: 4}
---

Ces deux méthodes changent le comportement d'un modèle **sans boucle d'entraînement** : pas de données annotées, pas d'heures de GPU. Elles répondent à deux besoins différents.

## Réunir plusieurs modèles spécialisés en un seul

**La situation.** On a affiné le même modèle de base plusieurs fois, chaque fois pour un usage : un modèle pour le support client, un pour rédiger du code, un pour le français juridique. On voudrait un seul modèle qui sache tout faire, plutôt que d'en servir trois.

**La méthode : la fusion.** Comme ces modèles partent des mêmes poids, chacun n'est qu'un petit déplacement par rapport à la base. La fusion additionne ces déplacements, en gardant les plus importants et en résolvant les conflits entre eux (méthodes TIES, DARE). Elle prend quelques minutes, tourne sur CPU, et ne demande aucune donnée.

**Les limites.**

- Les modèles doivent **venir du même modèle de base** : deux modèles d'origines différentes n'ont pas des poids comparables. Dans ce cas, la fusion est impossible et l'on sert les modèles séparément, ou l'on route chaque demande vers le bon.
- Si **une compétence se perd** dans le modèle fusionné, les spécialisations se gênent : il faut entraîner un seul modèle sur l'ensemble des données d'origine.

## Changer un trait du modèle

**La situation.** Le modèle fait ce qu'on attend, mais on voudrait ajuster sa manière de répondre : plus formel, plus positif, plus prudent, plus concis. On n'a pas de jeu d'exemples annotés pour l'entraîner.

**La méthode : le pilotage par activations.** On écrit quelques dizaines de paires de textes qui ne diffèrent que par ce trait (« Merci pour votre message, nous revenons vers vous. » / « Ouais, on verra. »). La différence moyenne des activations du modèle entre les deux côtés donne une direction ; on l'ajoute pendant la génération, avec une intensité réglable, sans toucher aux poids.

**Les limites.** Trop forte, la direction dégrade le texte ou déforme le sens ; trop faible, elle ne change rien. Si l'**effet est instable**, ou si l'on veut un changement durable inscrit dans le modèle, on apprend le trait pour de bon par **DPO**, sur ces mêmes paires présentées comme « réponse préférée / réponse rejetée ».
