---
title: Statut des exemples
id: statut-des-exemples
catalogue: true
---

Chaque exemple de code porte un statut honnête. Un exemple **démontré** traite un besoin réel, part d'un modèle qui échoue sur ce besoin, et mesure un gain net dans les mêmes conditions avant et après. Un exemple **non prouvé** s'exécute, mais son résultat ne montre pas encore le gain de la méthode : la raison est écrite à côté. Un exemple **prêt à tester** a le code de sa démonstration écrit, mais le calcul sur le GPU local n'a pas encore tourné. Les versions grosse infra (dossiers `infra/`) n'ont jamais été exécutées.

## Ce qu'exige une preuve

| Exigence | Mise en œuvre |
|---|---|
| Mesure avant / après dans les mêmes conditions | Même jeu de test, même décodage, même précision numérique, réponses assez longues pour ne pas être coupées |
| Jeu de test assez grand | Plusieurs centaines d'exemples ; chaque exactitude affichée avec son intervalle de confiance à 95 % (Wilson) |
| Pas un coup de chance | Comparaison appariée sur les mêmes questions : l'intervalle de la différence exclut 0 (bootstrap, test de McNemar) ; plusieurs graines (seeds) pour les entraînements courts (`make graines`) |
| Effet attribué à la méthode | Un témoin (baseline) quand une autre cause est possible : même entraînement sans la modification, même modèle sans RAG ou sans outils |
| Pas de fuite | Jeu de test jamais vu à l'entraînement |
| Non-régression | Une mesure hors de la tâche visée quand la méthode modifie les poids |
