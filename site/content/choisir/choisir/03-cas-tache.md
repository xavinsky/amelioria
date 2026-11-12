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
        récompense calculable: {m: 5}
---
