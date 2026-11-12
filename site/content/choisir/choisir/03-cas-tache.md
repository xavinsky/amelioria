---
title: "Cas : tâche, format ou style"
id: cas-tache
description: "Apprendre une tâche, un format ou un style à un modèle d'IA : prompt, LoRA, SFT complet, DPO, RL."
decision:
  q: "Que sait-on fournir\ncomme exemples ?"
  choix:
    Quelques-uns:
      m: 17
      choix:
        ne suffit pas: {fin: "Annoter des centaines\nd'exemples, puis LoRA"}
    Des centaines:
      m: 3
      choix:
        plafonne: {m: 2}
    Jugements A > B:
      m: 4
      choix:
        un programme note: {m: 5}
---

La question décisive est ce qu'on sait fournir pour montrer ce qu'on attend : quelques exemples, des centaines, ou seulement des jugements entre deux réponses.

## Quelques exemples : le prompt

Une consigne précise et quelques exemples dans le prompt suffisent souvent à fixer un format, un ton ou une tâche simple. Rien n'est entraîné, et la méthode marche avec un modèle fermé. Si le résultat **ne suffit pas**, il faut annoter des centaines d'exemples et passer à l'entraînement.

## Des centaines d'exemples : LoRA, puis SFT complet

Avec quelques centaines à quelques milliers d'exemples annotés, une [LoRA](#peft) apprend la tâche en n'entraînant qu'un petit module : peu de mémoire, peu de temps, peu d'oubli. Elle **plafonne** quand la tâche demande un changement profond du modèle ; le **SFT complet**, qui entraîne tous les poids, va alors plus loin, au prix de plusieurs GPU et d'un mélange de données générales pour limiter l'oubli.

## Seulement des jugements : DPO, puis RL

Quand on sait dire laquelle de deux réponses est la meilleure sans savoir écrire la bonne (ton, utilité, sécurité), le **DPO** apprend directement de ces paires. Si **un programme peut noter** les réponses (tests, règles, juge automatique), le **RL** fait répondre le modèle, note chaque réponse et le pousse vers les mieux notées : plus coûteux, mais il dépasse les exemples fournis.
