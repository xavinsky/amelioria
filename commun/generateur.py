"""Générateur de chat unique pour les méthodes qui ne modifient pas les poids (17 à 20).

Même interface pour un modèle local (transformers) et pour une API compatible OpenAI
(vLLM, Ollama, llama.cpp server, fournisseurs cloud) : ces méthodes s'appliquent aussi aux modèles fermés.

    gen = Generateur.depuis_config(cfg.modele)
    gen.chat([[{"role": "user", "content": "..."}], ...], max_new_tokens=64)          -> list[str]
    gen.chat(conversations, n=8, temperature=0.8)                                     -> list[list[str]]
"""

from __future__ import annotations

import json
import os
import urllib.request

from . import modeles


class Generateur:
    def __init__(self, nom: str, api_base: str | None = None, api_cle: str | None = None, dtype: str = "auto"):
        self.nom, self.api_base = nom, api_base
        self.api_cle = api_cle or os.environ.get("OPENAI_API_KEY", "x")
        self.model = self.tok = None
        if not api_base:
            self.model, self.tok = modeles.charger_llm(nom, dtype)

    @classmethod
    def depuis_config(cls, m):
        return cls(m.nom, m.get("api_base"), m.get("api_cle"), m.get("dtype", "auto"))

    def _texte(self, messages: list[dict]) -> str:
        if modeles.a_chat_template(self.tok):
            return self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        lignes = []
        for msg in messages:
            prefixe = {"system": "", "user": "Question: ", "assistant": "Answer: ", "tool": "Result: "}[msg["role"]]
            lignes.append(prefixe + msg["content"])
        return "\n".join(lignes) + "\nAnswer:"

    def chat(self, conversations: list[list[dict]], max_new_tokens: int = 128, temperature: float = 0.0, n: int = 1,
             batch: int = 16, stop: list[str] | None = None):
        if self.api_base:
            return [self._api(c, max_new_tokens, temperature, n, stop) for c in conversations]
        sorties = modeles.generer(self.model, self.tok, [self._texte(c) for c in conversations], max_new_tokens=max_new_tokens,
                                  temperature=temperature, n=n, batch=batch, **({"stop_strings": stop, "tokenizer": self.tok} if stop else {}))
        return sorties

    def _api(self, messages, max_tokens, temperature, n, stop):
        corps = {"model": self.nom, "messages": messages, "max_tokens": max_tokens, "temperature": temperature, "n": n}
        if stop:
            corps["stop"] = stop
        req = urllib.request.Request(f"{self.api_base.rstrip('/')}/chat/completions", json.dumps(corps).encode(),
                                     {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_cle}"})
        with urllib.request.urlopen(req, timeout=600) as rep:
            choix = [c["message"]["content"] or "" for c in json.load(rep)["choices"]]
        return choix[0] if n == 1 else choix
