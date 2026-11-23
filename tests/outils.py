"""Chemins et petit modèle aléatoire communs aux tests (CPU, sans téléchargement)."""

import pathlib
import sys

RACINE = pathlib.Path(__file__).resolve().parent.parent
for d in (RACINE, RACINE / "08-fusion-modeles", RACINE / "14-croissance-architecture", RACINE / "16-elagage"):
    sys.path.insert(0, str(d))

import torch  # noqa: E402
from transformers import Qwen2Config, Qwen2ForCausalLM  # noqa: E402


def petit_modele(couches: int = 4, graine: int = 0):
    torch.manual_seed(graine)
    cfg = Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=couches,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64)
    return Qwen2ForCausalLM(cfg).eval()
