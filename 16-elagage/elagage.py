"""Algorithmes d'élagage, en PyTorch pur, pour les LLM de la famille LLaMA / Qwen / Mistral / SmolLM.

Non structuré (zéros épars) : magnitude, wanda, 2:4 (semi-structuré, accéléré sur GPU Ampere+).
Structuré (le modèle rétrécit vraiment) : couches (ShortGPT), largeur (neurones MLP, recette Minitron).
"""

from __future__ import annotations

import torch
from transformers import TrainerCallback

from commun import modeles

NOMS_LINEAIRES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def lineaires(model):
    """Couches linéaires des blocs ; avec un adaptateur LoRA, la couche d'origine (`base_layer`) est celle qui compte."""
    if hasattr(model, "get_base_model"):
        model = model.get_base_model()
    for i, couche in enumerate(modeles.couches_decodeur(model)):
        for nom, mod in couche.named_modules():
            chemin = nom.split(".")
            if isinstance(mod, torch.nn.Linear) and (chemin[-1] in NOMS_LINEAIRES or (chemin[-1] == "base_layer" and chemin[-2] in NOMS_LINEAIRES)):
                yield f"{i}.{nom}", mod


@torch.no_grad()
def normes_entrees(model, tok, textes, longueur=256):
    """‖X_j‖₂ de chaque colonne d'entrée de chaque couche linéaire, sur des données de calibration (pour Wanda)."""
    normes, poignees = {}, []
    for nom, mod in lineaires(model):
        def capter(m, entrees, nom=nom):
            x = entrees[0].reshape(-1, entrees[0].shape[-1]).float()
            normes[nom] = normes.get(nom, 0) + (x**2).sum(0)
        poignees.append(mod.register_forward_pre_hook(capter))
    try:
        for t in textes:
            model(**tok(t, return_tensors="pt", truncation=True, max_length=longueur).to(model.device))
    finally:
        for p in poignees:
            p.remove()
    return {k: v.sqrt() for k, v in normes.items()}


@torch.no_grad()
def masque(score: torch.Tensor, sparsite: float, motif: str | None) -> torch.Tensor:
    """1 = poids conservé. Comparaison par ligne (neurone de sortie), comme Wanda."""
    if motif == "2:4":
        assert score.shape[1] % 4 == 0, "2:4 exige un nombre de colonnes multiple de 4"
        s = score.reshape(-1, 4)   # groupes de 4 poids consécutifs dans chaque ligne
        garde = torch.zeros_like(s, dtype=torch.bool).scatter_(1, s.topk(2, dim=1).indices, True)
        return garde.reshape(score.shape)
    k = int(score.shape[1] * sparsite)
    if k == 0:
        return torch.ones_like(score, dtype=torch.bool)
    seuils = score.kthvalue(k, dim=1, keepdim=True).values
    return score > seuils


@torch.no_grad()
def elaguer_non_structure(model, methode: str, sparsite: float, normes=None, motif=None) -> dict:
    """Met à zéro les poids élagués et renvoie les masques booléens (1 = conservé), à réappliquer pendant la cicatrisation."""
    masques = {}
    for nom, mod in lineaires(model):
        W = mod.weight.detach().float()
        score = W.abs() if methode == "magnitude" else W.abs() * normes[nom][None, :]
        masques[nom] = masque(score, sparsite, motif).to(mod.weight.device)
        mod.weight.mul_(masques[nom])
    return masques


class GarderMasques(TrainerCallback):
    """Réapplique les masques après chaque pas d'optimisation : les zéros restent des zéros pendant la cicatrisation.

    Masques booléens : 1 octet par poids, ce qui laisse la mémoire à l'entraînement complet.
    """

    def __init__(self, model, masques):
        self.modules = dict(lineaires(model))
        self.masques = masques

    @torch.no_grad()
    def on_step_end(self, args, state, control, **kwargs):
        for nom, m in self.masques.items():
            self.modules[nom].weight.mul_(m)


def sparsite_reelle(model) -> float:
    total = zeros = 0
    for _, mod in lineaires(model):
        total += mod.weight.numel()
        zeros += (mod.weight == 0).sum().item()
    return round(zeros / total, 4)


@torch.no_grad()
def importance_couches(model, tok, textes, longueur=256):
    """ShortGPT « Block Influence » : 1 - cos(entrée, sortie) du bloc. Faible = le bloc change peu la représentation."""
    couches = modeles.couches_decodeur(model)
    scores = torch.zeros(len(couches))
    poignees = []
    for i, c in enumerate(couches):
        def capter(m, args, kwargs, sortie, i=i):
            h_in = args[0] if args else kwargs["hidden_states"]
            h_out = sortie[0] if isinstance(sortie, tuple) else sortie
            scores[i] += (1 - torch.nn.functional.cosine_similarity(h_in.float(), h_out.float(), dim=-1)).mean().item()
        poignees.append(c.register_forward_hook(capter, with_kwargs=True))
    try:
        for t in textes:
            model(**tok(t, return_tensors="pt", truncation=True, max_length=longueur).to(model.device))
    finally:
        for p in poignees:
            p.remove()
    return scores / len(textes)


def supprimer_couches(model, indices: list[int]):
    couches = modeles.couches_decodeur(model)
    gardees = [c for i, c in enumerate(couches) if i not in set(indices)]
    model.model.layers = torch.nn.ModuleList(gardees)
    for i, c in enumerate(gardees):
        if hasattr(c.self_attn, "layer_idx"):
            c.self_attn.layer_idx = i
    model.config.num_hidden_layers = len(gardees)
    if getattr(model.config, "layer_types", None):
        model.config.layer_types = model.config.layer_types[: len(gardees)]


@torch.no_grad()
def importance_neurones(model, tok, textes, longueur=256):
    """Importance de chaque neurone intermédiaire des MLP : moyenne de |act(gate) * up| (entrée de down_proj)."""
    scores, poignees = {}, []
    for i, c in enumerate(modeles.couches_decodeur(model)):
        def capter(m, entrees, i=i):
            scores[i] = scores.get(i, 0) + entrees[0].float().abs().reshape(-1, entrees[0].shape[-1]).mean(0)
        poignees.append(c.mlp.down_proj.register_forward_pre_hook(capter))
    try:
        for t in textes:
            model(**tok(t, return_tensors="pt", truncation=True, max_length=longueur).to(model.device))
    finally:
        for p in poignees:
            p.remove()
    return scores


@torch.no_grad()
def reduire_largeur(model, scores: dict, ratio: float):
    """Garde les (1 - ratio) neurones les plus importants de chaque MLP : gate/up perdent des lignes, down des colonnes."""
    nouvelle_taille = None
    for i, c in enumerate(modeles.couches_decodeur(model)):
        k = int(c.mlp.down_proj.in_features * (1 - ratio))
        garde = scores[i].topk(k).indices.sort().values.to(c.mlp.down_proj.weight.device)
        for nom in ("gate_proj", "up_proj"):
            ancien = getattr(c.mlp, nom)
            nouveau = torch.nn.Linear(ancien.in_features, k, bias=ancien.bias is not None, device=ancien.weight.device, dtype=ancien.weight.dtype)
            nouveau.weight.copy_(ancien.weight[garde])
            if ancien.bias is not None:
                nouveau.bias.copy_(ancien.bias[garde])
            setattr(c.mlp, nom, nouveau)
        ancien = c.mlp.down_proj
        nouveau = torch.nn.Linear(k, ancien.out_features, bias=ancien.bias is not None, device=ancien.weight.device, dtype=ancien.weight.dtype)
        nouveau.weight.copy_(ancien.weight[:, garde])
        if ancien.bias is not None:
            nouveau.bias.copy_(ancien.bias)
        c.mlp.down_proj = nouveau
        nouvelle_taille = k
    model.config.intermediate_size = nouvelle_taille


@torch.no_grad()
def vitesse_semi_structuree(model, tok, tokens: int = 32, repetitions: int = 3, lot: int = 8, longueur: int = 512) -> dict:
    """Vitesse réelle du motif 2:4 : le même modèle en fp16 dense, puis ses couches linéaires converties en tenseurs
    semi-structurés (noyaux creux de PyTorch, cuSPARSELt). Modifie `model` : à appeler en dernier.

    Deux mesures : la génération d'une séquence (limitée par la lecture des poids) et le traitement d'un lot de
    `lot` x `longueur` tokens (limité par le calcul, là où le motif 2:4 peut accélérer). Si les noyaux ne sont pas
    disponibles sur la carte, renvoie l'erreur au lieu de lever."""
    import time

    from torch.sparse import to_sparse_semi_structured

    if hasattr(model, "get_base_model"):
        model = model.get_base_model()
    dev = next(model.parameters()).device
    if dev.type != "cuda":
        return {"erreur": "noyaux creux : GPU requis"}
    model.half().eval()
    ids = tok("The history of the city", return_tensors="pt").to(dev)
    bloc = torch.randint(0, model.config.vocab_size, (lot, longueur), device=dev)

    def chronometrer(f):
        f()   # préchauffage
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(repetitions):
            f()
        torch.cuda.synchronize()
        return (time.time() - t0) / repetitions

    def mesurer_les_deux():
        g = chronometrer(lambda: model.generate(**ids, max_new_tokens=tokens, min_new_tokens=tokens, do_sample=False))
        p = chronometrer(lambda: model(input_ids=bloc))
        return {"generation_tokens_par_s": round(tokens / g, 1), "traitement_tokens_par_s": round(lot * longueur / p)}

    res = {"dense_fp16": mesurer_les_deux()}
    try:
        for _, mod in lineaires(model):
            mod.weight = torch.nn.Parameter(to_sparse_semi_structured(mod.weight.data), requires_grad=False)
        res["semi_structure_2_4"] = mesurer_les_deux()
        res["gain_traitement"] = round(res["semi_structure_2_4"]["traitement_tokens_par_s"] / res["dense_fp16"]["traitement_tokens_par_s"], 3)
        res["gain_generation"] = round(res["semi_structure_2_4"]["generation_tokens_par_s"] / res["dense_fp16"]["generation_tokens_par_s"], 3)
    except Exception as e:   # noyaux absents, forme ou carte non prise en charge
        res["erreur"] = f"{type(e).__name__}: {e}"[:300]
    return res
