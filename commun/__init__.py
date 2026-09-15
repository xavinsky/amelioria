"""Socle partagé par tous les exemples : configuration, modèles, données, évaluation, rapport."""

import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRL_EXPERIMENTAL_SILENCE", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
