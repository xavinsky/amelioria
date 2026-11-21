# Améliorer un modèle d'IA sans le réentraîner de zéro

Panorama exécutable des méthodes d'amélioration d'un modèle existant : 20 dossiers, chacun avec un exemple local, des variantes et une version grosse infra. Chaque exemple doit traiter **un besoin réellement utile en entreprise**, avec **le bon concept**, une **implémentation limpide**, et un **résultat prouvé sans ambiguïté** : intervalle de confiance, plusieurs graines, témoin, non-régression. Pas de démonstration fabriquée ni de réglage ad hoc pour obtenir le chiffre voulu. Si le résultat n'apparaît pas, l'approche ou l'implémentation est mauvaise : relire les papiers de la méthode et se réorienter. Code, configs et docs décrivent l'état actuel, jamais leur historique.

## GPU local

La règle d'usage du GPU est dans `~/.claude/CLAUDE.md` (« GPU local ») : tout passe par le skill `gpu-local`.

Dans ce projet, sollicitent le GPU : l'exécution des exemples de chaque méthode — pré-entraînement continu, fine-tuning, PEFT, optimisation des préférences, RL, distillation, quantification, élagage, inférence — dès qu'un modèle est chargé avec PyTorch, transformers, vLLM ou Ollama. Un entraînement long se lance par `gpu-local executer`.
