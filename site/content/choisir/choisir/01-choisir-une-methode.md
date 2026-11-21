---
title: Quelle méthode pour quel besoin
id: choisir-une-methode
description: "Arbre de décision pour améliorer un modèle d'IA : quelle méthode choisir selon le besoin (connaissances, tâche, raisonnement, coût, contexte), avec le calcul et le temps de chacune."
arbre:
  racine: Que faut-il améliorer ?
  branches:
    - question: "Ses\nconnaissances"
      cas: cas-connaissances
      feuilles:
        - {si: "Ajouter des documents", m: 18}
        - {si: "Savoir à intégrer", m: 1, nom: "Pré-entraînement", apres: 18, note: "suivi d'un SFT pour savoir y répondre"}
        - {si: "Corriger quelques faits", m: 9, nom: "Édition"}
        - {si: "Retirer un savoir", m: 10, nom: "Désapprentissage"}
    - question: "Tâche, format\nou style"
      cas: cas-tache
      feuilles:
        - {si: "Consigne et exemples", m: 17, nom: "Prompt"}
        - {si: "Centaines d'exemples", m: 3, nom: "LoRA, QLoRA", apres: 17}
        - {si: "La LoRA plafonne", m: 2, nom: "SFT complet", apres: 3}
        - {si: "On sait juger A > B", m: 4, nom: "DPO"}
    - question: "Raisonnement\n(maths, code)"
      cas: cas-raisonnement
      feuilles:
        - {si: "Plusieurs essais, vote", m: 20, nom: "Inférence"}
        - {si: "Réponses vérifiables", m: 7, nom: "Données synth.", apres: 20}
        - {si: "Récompense calculable", m: 5, nom: "RL", apres: 7}
        - {si: "Calcul exact, actions", m: 19, nom: "Outils"}
    - question: "Sans\nréentraîner"
      cas: cas-sans-reentrainer
      feuilles:
        - {si: "Doser un trait (ton)", m: 11, nom: "Pilotage (steering)"}
        - {si: "Fusionner des fine-tunes", m: 8, nom: "Fusion"}
    - question: "Trop gros\nou trop lent"
      cas: cas-cout
      feuilles:
        - {si: "Même modèle, plus léger", m: 15, nom: "Quantification"}
        - {si: "Plus petit, réentraîné", m: 6, nom: "Distillation", apres: 15}
        - {si: "Couches superflues", m: 16, nom: "Élagage"}
    - question: "Limites de\nla forme"
      cas: cas-forme
      feuilles:
        - {si: "Textes > fenêtre", m: 12, nom: "Contexte"}
        - {si: "Langue mal découpée", m: 13, nom: "Vocabulaire"}
        - {si: "Plus de capacité", m: 14, nom: "Croissance"}
---

> [!TIP]
> Les méthodes de la catégorie EXEC « À l'exécution, sans toucher aux poids » marchent aussi avec un modèle fermé appelé par API. Elles se combinent avec toutes les autres : un modèle affiné se sert encore avec un bon prompt, du RAG et des outils.

## Recette d'un modèle post-entraîné

Les grands modèles ouverts (Tülu 3, Llama 3, Qwen) enchaînent plusieurs de ces méthodes. L'ordre type :

1. **Pré-entraînement continu** sur des données de qualité, éventuellement avec **extension de contexte**.
2. **SFT** sur des instructions, enrichi de **données synthétiques** filtrées.
3. **DPO** sur des préférences, dont des préférences produites par le modèle lui-même.
4. **RL à récompense vérifiable** sur les maths, le code et le respect des consignes.
5. **Fusion** éventuelle de plusieurs variantes.
6. **[Quantification](#quantification)** pour le service.
7. Au déploiement : prompt système, **[RAG](#rag)**, outils et calcul à l'inférence.
