# Cahier des charges des exemples

Ce que chaque dossier de méthode doit démontrer, et comment le prouver.

## Objectif

Pour chaque méthode d'amélioration d'un modèle : **un problème réellement rencontré en entreprise**, **le concept adapté à ce problème**, **une implémentation propre et lisible**, et **un résultat mesuré sans ambiguïté**. Le lecteur doit pouvoir transposer l'exemple à son propre cas en changeant les données et le modèle.

## Critères

### 1. Un vrai besoin

- Le problème est formulé comme un besoin métier : router des demandes clients, extraire des champs d'un document, répondre sur une base documentaire interne, réduire le coût de service, retirer une donnée personnelle, corriger un fait, etc.
- Les données sont un jeu public reconnu, ou un jeu généré dont la construction est explicite et qui reproduit une situation réelle (par exemple une base de connaissances que le modèle ne peut pas connaître).
- Le modèle de départ doit **échouer de façon mesurable** sur le besoin. Si le modèle résout déjà la tâche, l'exemple ne démontre rien : changer de tâche ou de modèle, pas de réglages.

### 2. Le bon concept

- La méthode choisie est celle qu'on recommanderait pour ce besoin (voir [choisir-une-methode.md](choisir-une-methode.md)).
- La recette suit les papiers de référence de la méthode ([references.md](references.md)) : données, perte (loss), hyperparamètres typiques. Quand un écart est nécessaire (taille du modèle, budget), il est dit dans le README.

### 3. Une implémentation limpide

- Un `run.py` lisible de bout en bout : mesurer, appliquer la méthode, remesurer, sauvegarder.
- Pas de réglage ad hoc qui n'aurait de sens que pour obtenir le chiffre voulu.
- Les commentaires expliquent le concept ou la raison d'un choix, jamais l'histoire du code.

### 4. Une preuve sans ambiguïté

| Exigence | Mise en œuvre |
|---|---|
| Mesure avant / après dans les mêmes conditions | Même jeu de test, même décodage, même précision numérique, même longueur de génération suffisante pour ne pas couper les réponses |
| Jeu de test assez grand | Plusieurs centaines d'exemples ; intervalle de confiance à 95 % affiché pour les exactitudes |
| Pas un coup de chance | Écart supérieur à l'incertitude ; pour les entraînements courts, plusieurs graines (seeds) (moyenne et écart) |
| Effet attribué à la méthode | Témoin (baseline) quand une autre cause est possible : même entraînement sans la modification (FORM13, FORM14), même élève sans enseignant (SIGN06), même modèle sans outils (EXEC19) ou sans RAG (EXEC18) |
| Pas de fuite | Jeu de test jamais vu à l'entraînement ; textes de perplexité sur un split distinct |
| Non-régression | Une mesure hors de la tâche visée (perplexité générale, tâche voisine) quand la méthode modifie les poids |
| Coût | Durée, mémoire, paramètres entraînés, taille produite |

### 5. Documentation

- Section « Résultats mesurés » dans chaque README : tableau avant / après avec incertitude, matériel, durée, date de la mesure.
- Le README décrit l'état actuel de l'exemple, sans historique.

## Grille de validation d'un exemple

Un exemple est validé quand toutes les réponses sont « oui » :

1. Le besoin se comprend sans connaître la méthode ?
2. Le modèle de départ échoue sur ce besoin ?
3. La recette correspond aux papiers de référence ?
4. Le gain dépasse l'intervalle de confiance, et se reproduit avec une autre graine ?
5. Un témoin exclut les autres explications, quand il y en a ?
6. La non-régression est mesurée ?
7. Le code se lit sans connaître son historique ?
