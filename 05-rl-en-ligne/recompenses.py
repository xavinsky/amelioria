"""Sources de récompense pour le RL en ligne. Signature TRL : f(prompts, completions, **colonnes) -> list[float].

- regles  : vérificateur programmatique (RLVR) — exactitude de la réponse, respect du format.
- modele  : modèle de récompense entraîné sur des préférences (RLHF) — géré nativement par TRL (chemin/nom).
- juge    : LLM juge qui note la réponse (RLAIF) — modèle local transformers ou API compatible OpenAI (vLLM, Ollama...).
"""

from __future__ import annotations

import json
import pathlib
import re
import sys
import urllib.request

sys.path.insert(0, str(next(p for p in pathlib.Path(__file__).resolve().parents if (p / "commun").is_dir())))

from commun import donnees, modeles


def _texte(c) -> str:
    return c[-1]["content"] if isinstance(c, list) else c


def exactitude(prompts, completions, cible, **_):
    """1 si la réponse finale est juste, 0 sinon : la récompense « vérifiable » par excellence."""
    return [1.0 if donnees.extraire_nombre(_texte(c)) == str(t) else 0.0 for c, t in zip(completions, cible)]


def format_reponse(prompts, completions, **_):
    """Petit bonus si la réponse se termine par « Answer: <nombre> » : guide l'exploration au début."""
    return [0.2 if re.search(r"Answer:\s*-?\d+\s*$", _texte(c).strip()) else 0.0 for c in completions]


def fonctions_regles():
    return [exactitude, format_reponse]


CONSIGNE_JUGE = (
    "You are a strict grader. Rate the assistant's answer to the user's request on a scale from 1 to 10 "
    "for correctness, helpfulness and concision. Reply with the number only.\n\n"
    "Request:\n{prompt}\n\nAnswer:\n{reponse}\n\nScore:"
)


def fonction_juge(r):
    """Construit une fonction de récompense « LLM juge » (RLAIF)."""
    if r.get("api_base"):
        def appeler(textes):
            notes = []
            for t in textes:
                corps = json.dumps({"model": r.juge, "messages": [{"role": "user", "content": t}], "max_tokens": 4, "temperature": 0}).encode()
                req = urllib.request.Request(f"{r.api_base}/chat/completions", corps, {"Content-Type": "application/json",
                                             "Authorization": f"Bearer {r.get('api_cle', 'x')}"})
                with urllib.request.urlopen(req, timeout=120) as rep:
                    notes.append(json.load(rep)["choices"][0]["message"]["content"])
            return notes
    else:
        juge, tok = modeles.charger_llm(r.juge)
        juge.requires_grad_(False)

        def appeler(textes):
            prompts = [modeles.prompt_texte(tok, t) for t in textes]
            return modeles.generer(juge, tok, prompts, max_new_tokens=4, batch=16)

    def recompense_juge(prompts, completions, **_):
        textes = [CONSIGNE_JUGE.format(prompt=_texte(p), reponse=_texte(c)) for p, c in zip(prompts, completions)]
        notes = []
        for sortie in appeler(textes):
            m = re.search(r"\d+", sortie)
            notes.append(min(int(m.group()), 10) / 10 if m else 0.0)
        return notes

    return recompense_juge


def construire(r):
    """Section `recompense` de la config -> liste passée à GRPOTrainer(reward_funcs=...)."""
    if r.source == "regles":
        return fonctions_regles()
    if r.source == "modele":
        return [r.modele]   # TRL charge un AutoModelForSequenceClassification et son tokenizer
    if r.source == "juge":
        return [fonction_juge(r)]
    raise ValueError(f"source de récompense inconnue : {r.source}")
