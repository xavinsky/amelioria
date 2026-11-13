---
title: "AmeliorIA : améliorer un modèle d'IA existant"
nav: Présentation du projet
id: presentation
description: "AmeliorIA : panorama exécutable des méthodes pour améliorer un modèle d'IA existant sans le réentraîner de zéro, avec un exemple par méthode et un statut honnête de chaque résultat."
compteurs: true
---

Vous avez un modèle d'IA, ouvert ou accessible par API, et il ne fait pas tout à fait ce que vous voulez : il ignore vos documents, se trompe sur votre tâche, raisonne mal, coûte trop cher à servir. **AmeliorIA présente les méthodes pour l'améliorer sans le réentraîner de zéro**, et aide à choisir la bonne.

> [!WARNING]
> **Projet en cours.** **{{demontres}} exemples sur {{exemples}}** démontrent déjà, en local, un gain net de leur méthode. Les {{non_prouves}} autres s'exécutent, mais leur résultat ne prouve pas encore ce gain : le travail en cours consiste à les refondre sur des tâches réelles pour finir de les démontrer en local. Les {{infra}} versions « grosse infra », qui demandent plusieurs GPU de centre de calcul, sont en attente d'une infra cloud : leur code est fourni, mais n'a jamais été exécuté. Le détail, test par test : [Travail en cours](#travail-en-cours).
>
> **Vous pouvez prêter des GPU ?** Pour tester ces versions grosse infra, je cherche un accès à une machine multi-GPU (idéalement un nœud de 8 GPU de 80 Go, même pour quelques heures). Si vous pouvez en fournir une, je suis preneur : [contactez-moi sur GitHub](https://github.com/xavinsky) !

## Ce que propose le site

- **[Choisir une méthode](#choisir-une-methode)** : un arbre de décision qui part de ce qui ne va pas, puis six cas détaillés. Chaque méthode y affiche son coût (calcul, temps) et ses contraintes (accès au modèle, données à fournir).
- **[Les méthodes](#choisir-data)** : {{methodes}} méthodes rangées en {{categories}} catégories, de l'entraînement sur vos données jusqu'aux améliorations qui ne touchent pas aux poids. Pour chacune, une page qui explique le concept et ses références, et une page qui décrit l'exemple du dépôt : composants, étapes, résultats mesurés, commandes.
- **[Le code](#installer)** : un dépôt où chaque méthode a un exemple exécutable sur un seul GPU de 12 Go, avec les mêmes commandes partout (`make test`, `make run`).
- **[Le lexique](#lexique)** : les sigles et les noms de modèles, d'algorithmes et d'outils cités.

## Les cinq catégories

| Code | Catégorie | En une phrase |
|---|---|---|
| **DATA** | [Apprendre à partir de données](#choisir-data) | Entraîner les poids sur vos textes ou vos exemples annotés. |
| **SIGN** | [Apprendre à partir d'un signal](#choisir-sign) | Apprendre d'une préférence, d'un vérificateur, d'une note ou d'un modèle enseignant. |
| **EDIT** | [Chirurgie du modèle](#choisir-edit) | Fusionner des modèles, corriger, effacer ou orienter un savoir précis. |
| **FORM** | [Remodeler : étendre ou compresser](#choisir-form) | Allonger le contexte, ajouter du vocabulaire, agrandir, quantifier, élaguer. |
| **EXEC** | [À l'exécution, sans toucher aux poids](#choisir-exec) | Prompt, RAG, outils, calcul à l'inférence : marche aussi avec un modèle fermé. |

## Des résultats honnêtes

Chaque exemple porte un statut : **démontré** (gain mesuré et net), **non prouvé** (s'exécute, gain pas encore établi, avec la raison écrite à côté) ou **non testé** (versions grosse infra). Une preuve demande un besoin réel, un modèle de départ qui échoue, une mesure avant et après dans les mêmes conditions, un intervalle de confiance et, quand une autre cause est possible, un témoin. Le détail de chaque exemple est dans [Statut des exemples](#statut-des-exemples).
