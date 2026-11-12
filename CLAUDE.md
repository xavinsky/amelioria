# Améliorer un modèle d'IA sans le réentraîner de zéro

Panorama exécutable des méthodes d'amélioration d'un modèle existant, un dossier par méthode.

## GPU local

La règle d'usage du GPU est dans `~/.claude/CLAUDE.md` (« GPU local ») : tout passe par le skill `gpu-local`.

Dans ce projet, sollicitent le GPU : l'exécution des exemples de chaque méthode — pré-entraînement continu, fine-tuning, PEFT, optimisation des préférences, RL, distillation, quantification, élagage, inférence — dès qu'un modèle est chargé avec PyTorch, transformers, vLLM ou Ollama. Un entraînement long se lance par `gpu-local executer`.
