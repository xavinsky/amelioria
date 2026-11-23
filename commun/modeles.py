"""Chargement de modèles et génération, pour n'importe quel LLM causal Hugging Face.

Tout passe par les classes `Auto*` : un modèle LLaMA, Qwen, Mistral, Gemma, Phi, SmolLM, GPT-2...
se charge de la même façon. Les modèles de base (sans chat template) sont formatés en
"Question: ...\\nAnswer:" ; les modèles instruct utilisent leur propre chat template.
"""

from __future__ import annotations

import contextlib
import os
import pathlib
import re
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def appareil() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def dtype_calcul() -> torch.dtype:
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float32


def bf16_actif() -> bool:
    return dtype_calcul() == torch.bfloat16


def _dtype(nom):
    if nom in (None, "auto"):
        return dtype_calcul()
    return {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[nom]


def charger_tokenizer(nom: str, **kw):
    tok = AutoTokenizer.from_pretrained(nom, **kw)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def config_quantification(quantification: str | None):
    """`4bit` (NF4, QLoRA), `8bit` (LLM.int8) ou None. Nécessite CUDA + bitsandbytes."""
    if not quantification:
        return None
    from transformers import BitsAndBytesConfig

    if quantification == "4bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype_calcul(),
        )
    if quantification == "8bit":
        return BitsAndBytesConfig(load_in_8bit=True)
    raise ValueError(f"quantification inconnue : {quantification}")


def charger_llm(nom: str, dtype: str | None = "auto", quantification: str | None = None, **kw):
    """Charge un LLM causal et son tokenizer, placés sur le meilleur appareil disponible."""
    tok = charger_tokenizer(nom)
    qconf = config_quantification(quantification)
    # Lancé par accelerate/torchrun sur plusieurs GPU : c'est le Trainer (FSDP, DeepSpeed) qui place le modèle.
    distribue = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if qconf is not None:
        if distribue:
            if quantification == "4bit":
                qconf.bnb_4bit_quant_storage = _dtype(dtype)   # requis pour découper un modèle 4 bits avec FSDP
        else:
            kw.setdefault("device_map", "auto")
        model = AutoModelForCausalLM.from_pretrained(nom, quantization_config=qconf, dtype=_dtype(dtype), **kw)
    else:
        model = AutoModelForCausalLM.from_pretrained(nom, dtype=_dtype(dtype), **kw)
        if not distribue and "device_map" not in kw:   # device_map="auto" : déjà réparti sur les GPU
            model = model.to(appareil())
    if model.generation_config.pad_token_id is None:
        model.generation_config.pad_token_id = tok.pad_token_id
    return model, tok


def a_chat_template(tok) -> bool:
    return bool(getattr(tok, "chat_template", None))


def messages(question: str, systeme: str | None = None, reponse: str | None = None) -> list[dict]:
    m = []
    if systeme:
        m.append({"role": "system", "content": systeme})
    m.append({"role": "user", "content": question})
    if reponse is not None:
        m.append({"role": "assistant", "content": reponse})
    return m


def prompt_texte(tok, question: str, systeme: str | None = None) -> str:
    """Prompt prêt pour la génération (chat template si disponible)."""
    if a_chat_template(tok):
        return tok.apply_chat_template(messages(question, systeme), tokenize=False, add_generation_prompt=True)
    prefixe = f"{systeme}\n\n" if systeme else ""
    return f"{prefixe}Question: {question}\nAnswer:"


def exemple_sft(tok, question: str, reponse: str, systeme: str | None = None) -> dict:
    """Exemple au format prompt/completion de TRL (perte calculée sur la réponse seulement)."""
    if a_chat_template(tok):
        return {
            "prompt": messages(question, systeme),
            "completion": [{"role": "assistant", "content": reponse}],
        }
    prefixe = f"{systeme}\n\n" if systeme else ""
    return {"prompt": f"{prefixe}Question: {question}\nAnswer:", "completion": f" {reponse}{tok.eos_token}"}


def sans_autocast(model):
    """Retire l'autocast qu'accelerate laisse sur `model.forward` après un entraînement en précision mixte.

    Sans ça, un modèle aux poids fp32 est mesuré en fp32 avant l'entraînement et en bf16 après :
    les deux mesures ne sont plus comparables.
    """
    for m in (model, getattr(model, "base_model", None)):
        if m is not None and "_original_forward" in m.__dict__:
            del m.__dict__["_original_forward"]
            m.__dict__.pop("forward", None)   # le forward de la classe redevient actif
    return model


@contextlib.contextmanager
def padding_gauche(tok):
    ancien = tok.padding_side
    tok.padding_side = "left"
    try:
        yield
    finally:
        tok.padding_side = ancien


@torch.no_grad()
def generer(model, tok, prompts: list[str], max_new_tokens: int = 64, temperature: float = 0.0,
            n: int = 1, top_p: float = 1.0, batch: int = 16, intervalle: float = 60.0, **gen_kw):
    """Génère pour une liste de prompts texte. Renvoie list[str] (n=1) ou list[list[str]].

    Au-delà de `intervalle` secondes d'une génération, affiche l'avancement : lots faits, temps écoulé et restant,
    longueur moyenne générée par lot et nombre de lots allés jusqu'à `max_new_tokens` (signe qu'une suite du lot
    ne s'arrête pas d'elle-même et que le lot entier coûte la longueur maximale)."""
    sans_autocast(model).eval()
    sorties: list[list[str]] = []
    dev = next(model.parameters()).device
    n_lots, debut, dernier, longueurs = -(-len(prompts) // batch), time.monotonic(), time.monotonic(), []
    with padding_gauche(tok):
        for i in range(0, len(prompts), batch):
            lot = prompts[i : i + batch]
            enc = tok(lot, return_tensors="pt", padding=True, add_special_tokens=not a_chat_template(tok)).to(dev)
            params = dict(max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id, num_return_sequences=n)
            if temperature > 0:
                params.update(do_sample=True, temperature=temperature, top_p=top_p)
            else:
                params.update(do_sample=False, temperature=None, top_p=None, top_k=None)
            if not a_chat_template(tok):
                from transformers import StoppingCriteriaList

                params["stopping_criteria"] = StoppingCriteriaList([_FinReponse(tok, enc["input_ids"].shape[1])])
            params.update(gen_kw)
            out = model.generate(**enc, **params)
            longueurs.append(out.shape[1] - enc["input_ids"].shape[1])
            textes = tok.batch_decode(out[:, enc["input_ids"].shape[1] :], skip_special_tokens=True)
            if not a_chat_template(tok):
                textes = [couper_reponse(t) for t in textes]
            for j in range(len(lot)):
                sorties.append(textes[j * n : (j + 1) * n])
            fait, maintenant = len(longueurs), time.monotonic()
            if maintenant - dernier >= intervalle or (fait == n_lots and maintenant - debut >= intervalle):
                dernier, ecoule = maintenant, maintenant - debut
                print(f"[generer] lot {fait}/{n_lots} | écoulé {ecoule / 60:.1f} min, reste ~{ecoule / fait * (n_lots - fait) / 60:.1f} min"
                      f" | longueur moyenne {sum(longueurs) / fait:.0f} tokens, {sum(x >= max_new_tokens for x in longueurs)} lots au maximum",
                      flush=True)
    return [s[0] for s in sorties] if n == 1 else sorties


def generer_vllm(modele: str, prompts: list[str], python: str, max_new_tokens: int = 64, temperature: float = 0.0, n: int = 1,
                 top_p: float = 1.0, graine: int = 0, texte_brut: bool = True, **reglages):
    """Même contrat que `generer` (list[str] pour n=1, sinon list[list[str]]), par vLLM dans un processus du venv `python`.

    `modele` est un nom du Hub ou un dossier de modèle sauvegardé : vLLM recharge les poids, le modèle en mémoire
    n'est pas utilisé. Chaque suite s'arrête à sa propre fin : en format texte (`texte_brut`), à « \\nQuestion: »."""
    import json, subprocess, tempfile

    with tempfile.TemporaryDirectory() as tmp:
        demande, reponse = f"{tmp}/demande.json", f"{tmp}/reponse.json"
        json.dump(dict(modele=modele, prompts=prompts, max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p,
                       n=n, graine=graine, stop=["\nQuestion:"] if texte_brut else None, **reglages),
                  open(demande, "w", encoding="utf-8"), ensure_ascii=False)
        script = pathlib.Path(__file__).with_name("vllm_generer.py")
        subprocess.run([python, str(script), demande, reponse], check=True)
        sorties = json.load(open(reponse, encoding="utf-8"))
    if texte_brut:
        sorties = [[couper_reponse(t) for t in s] for s in sorties]
    return [s[0] for s in sorties] if n == 1 else sorties


FIN_REPONSE = re.compile(r"Answer:[^\n]*\S[^\n]*\n")


class _FinReponse:
    """Arrête chaque suite à la fin de sa ligne « Answer: … » (format texte) : un modèle de base enchaîne
    sinon sur du texte sans rapport jusqu'à `max_new_tokens`, ce qui fausse l'extraction et coûte du calcul."""

    def __init__(self, tok, debut: int, fenetre: int = 64):
        self.tok, self.debut, self.fenetre = tok, debut, fenetre

    def __call__(self, input_ids, scores, **kw):
        textes = self.tok.batch_decode(input_ids[:, self.debut :][:, -self.fenetre :], skip_special_tokens=True)
        return torch.tensor([bool(FIN_REPONSE.search(t)) for t in textes], device=input_ids.device)


def couper_reponse(texte: str) -> str:
    """Texte généré au format « Question: … Answer: » ramené à la réponse : avant la question suivante,
    et jusqu'à la fin de la ligne « Answer: … »."""
    texte = texte.split("\nQuestion:")[0]
    m = re.search(r"Answer:[^\n]*", texte)
    return texte[: m.end()] if m else texte


@torch.no_grad()
def logprob_suites(model, tok, prefixes: list[str], suites: list[str]) -> torch.Tensor:
    """Somme des log-probabilités de chaque `suite` sachant son `prefixe`."""
    sans_autocast(model).eval()
    dev = next(model.parameters()).device
    res = []
    for pre, suite in zip(prefixes, suites):
        ids_pre = tok(pre, add_special_tokens=False)["input_ids"]
        ids_suite = tok(suite, add_special_tokens=False)["input_ids"]
        ids = torch.tensor([ids_pre + ids_suite], device=dev)
        logits = model(ids).logits[0, :-1].float()
        lp = torch.log_softmax(logits, -1)
        cibles = ids[0, 1:]
        debut = len(ids_pre) - 1
        res.append(lp[debut:].gather(-1, cibles[debut:, None]).sum().item())
    return torch.tensor(res)


def compter_parametres(model) -> dict:
    total = sum(p.numel() for p in model.parameters())
    entr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "entrainables": entr, "pourcentage_entrainable": round(100 * entr / max(total, 1), 4)}


def taille_dossier_mo(dossier) -> float:
    return round(sum(f.stat().st_size for f in pathlib.Path(dossier).rglob("*") if f.is_file()) / 2**20, 2)


def sauvegarder(model, tok, dossier) -> str:
    dossier = pathlib.Path(dossier)
    dossier.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(dossier)
    if tok is not None:
        tok.save_pretrained(dossier)
    print(f"[modele] sauvegardé dans {dossier} ({taille_dossier_mo(dossier)} Mo)")
    return str(dossier)


def couches_decodeur(model):
    """Liste des blocs transformer d'un LLM causal, quelle que soit la famille."""
    for chemin in ("model.layers", "model.decoder.layers", "transformer.h", "gpt_neox.layers", "model.model.layers"):
        obj = model
        try:
            for attr in chemin.split("."):
                obj = getattr(obj, attr)
            return obj
        except AttributeError:
            continue
    raise ValueError("structure de modèle non reconnue : impossible de trouver les couches")
