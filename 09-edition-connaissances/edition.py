"""ROME (Rank-One Model Editing) et FT-L (fine-tuning contraint d'une couche), implémentés directement.

Idée de ROME : la couche de sortie d'un MLP (`down_proj`) se comporte comme une mémoire clé -> valeur.
La clé k* est l'activation au dernier token du sujet (« Eiffel Tower ») ; on calcule la valeur v* qui fait
prédire le nouvel objet (« Rome »), puis on insère l'association par une mise à jour de rang 1 qui perturbe
le moins possible les autres clés (pondération par la covariance C des clés sur du texte générique) :

    W' = W + (v* - W k*) (C⁻¹ k*)ᵀ / ((C⁻¹ k*)ᵀ k*)
"""

from __future__ import annotations

import contextlib

import torch
import torch.nn.functional as F

CONTEXTES = ["{}", "The following is a well-known fact. {}", "I read something interesting today. {}",
             "Here is what the encyclopedia says. {}", "Let me tell you something. {}"]


def module(model, chemin: str, couche: int):
    obj = model
    for attr in chemin.format(couche).split("."):
        obj = getattr(obj, attr)
    return obj


@contextlib.contextmanager
def crochet(mod, avant=None, apres=None):
    poignees = []
    if avant:
        poignees.append(mod.register_forward_pre_hook(avant))
    if apres:
        poignees.append(mod.register_forward_hook(apres))
    try:
        yield
    finally:
        for p in poignees:
            p.remove()


def position_sujet(tok, texte: str, sujet: str) -> int:
    """Indice du dernier token du sujet dans `texte`."""
    fin = texte.index(sujet) + len(sujet)
    return len(tok(texte[:fin], add_special_tokens=False)["input_ids"]) - 1


@torch.no_grad()
def covariance_cles(model, tok, mod, textes: list[str], longueur_max: int = 256) -> torch.Tensor:
    """C = E[k kᵀ] des entrées de `mod` sur du texte générique."""
    somme, n = None, 0

    def capter(_, entrees):
        nonlocal somme, n
        k = entrees[0].reshape(-1, entrees[0].shape[-1]).float()
        somme = k.T @ k if somme is None else somme + k.T @ k
        n += k.shape[0]

    with crochet(mod, avant=capter):
        for t in textes:
            model(**tok(t, return_tensors="pt", truncation=True, max_length=longueur_max).to(model.device))
    return somme / n


@torch.no_grad()
def cle_sujet(model, tok, mod, prompt: str, sujet: str) -> torch.Tensor:
    """k* : entrée de `mod` au dernier token du sujet, moyennée sur plusieurs contextes."""
    cles = []
    for ctx in CONTEXTES:
        texte = ctx.format(prompt)
        pos = position_sujet(tok, texte, sujet)

        def capter(_, entrees, pos=pos):
            cles.append(entrees[0][0, pos].float())

        with crochet(mod, avant=capter):
            model(**tok(texte, return_tensors="pt", add_special_tokens=False).to(model.device))
    return torch.stack(cles).mean(0)


def valeur_cible(model, tok, mod, edit: dict, p) -> torch.Tensor:
    """v* : sortie de `mod` au token du sujet qui maximise log P(nouvel objet), avec régularisation KL sur le sens du sujet."""
    dev = model.device
    ids_objet = tok(edit["nouvelle"], add_special_tokens=False)["input_ids"]
    textes = [ctx.format(edit["prompt"]) for ctx in CONTEXTES]
    essence = f"{edit['sujet']} is a"
    lots, positions, debuts = [], [], []
    for t in textes + [essence]:
        ids = tok(t, add_special_tokens=False)["input_ids"]
        positions.append(position_sujet(tok, t, edit["sujet"]))
        debuts.append(len(ids))
        lots.append(ids + (ids_objet if t != essence else []))

    longueur = max(len(x) for x in lots)
    entree = torch.full((len(lots), longueur), tok.pad_token_id, device=dev)
    attention = torch.zeros_like(entree)
    for i, x in enumerate(lots):
        entree[i, : len(x)] = torch.tensor(x, device=dev)
        attention[i, : len(x)] = 1

    with torch.no_grad():
        valeurs = {}

        def capter(_, __, sortie):
            valeurs["v"] = sortie[0, positions[0]].float().clone()

        with crochet(mod, apres=capter):
            logits = model(input_ids=entree, attention_mask=attention).logits
        v_orig = valeurs["v"]
        ref_essence = F.log_softmax(logits[-1, debuts[-1] - 1].float(), -1)

    delta = torch.zeros_like(v_orig, requires_grad=True)
    optim = torch.optim.Adam([delta], lr=p.lr_valeur)
    for _ in range(p.pas_valeur):
        def injecter(_, __, sortie):
            sortie = sortie.clone()
            for i, pos in enumerate(positions):
                sortie[i, pos] = sortie[i, pos] + delta.to(sortie.dtype)
            return sortie

        with crochet(mod, apres=injecter):
            logits = model(input_ids=entree, attention_mask=attention).logits.float()
        lp = F.log_softmax(logits, -1)
        nll = 0.0
        for i in range(len(textes)):
            for j, tid in enumerate(ids_objet):
                nll = nll - lp[i, debuts[i] - 1 + j, tid]
        nll = nll / (len(textes) * len(ids_objet))
        kl = F.kl_div(lp[-1, debuts[-1] - 1], ref_essence, log_target=True, reduction="sum")
        perte = nll + p.poids_kl * kl + p.poids_norme * (delta.norm() / v_orig.norm()) ** 2
        optim.zero_grad()
        perte.backward()
        optim.step()
        with torch.no_grad():   # borne la taille de la modification
            maxi = p.norme_max * v_orig.norm()
            if delta.norm() > maxi:
                delta.mul_(maxi / delta.norm())
        if nll.item() < 0.05:
            break
    return (v_orig + delta).detach()


@torch.no_grad()
def rome(model, tok, edit: dict, p, C_inv: torch.Tensor) -> dict:
    mod = module(model, p.module, p.couche)
    k = cle_sujet(model, tok, mod, edit["prompt"], edit["sujet"])
    with torch.enable_grad():
        v = valeur_cible(model, tok, mod, edit, p)
    W = mod.weight
    u = C_inv @ k
    residu = v - W.float() @ k
    W.add_((torch.outer(residu, u) / (u @ k)).to(W.dtype))
    return {"norme_mise_a_jour": round((residu.norm() * u.norm() / (u @ k).abs()).item(), 4)}


def ft_contraint(model, tok, edit: dict, p) -> dict:
    """FT-L : fine-tuning de la seule matrice ciblée, avec ‖ΔW‖∞ ≤ ε (référence simple, souvent moins spécifique que ROME)."""
    mod = module(model, p.module, p.couche)
    W0 = mod.weight.detach().clone()
    for prm in model.parameters():
        prm.requires_grad_(False)
    mod.weight.requires_grad_(True)
    optim = torch.optim.Adam([mod.weight], lr=p.lr_ft)
    ids_p = tok(edit["prompt"], add_special_tokens=False)["input_ids"]
    ids_o = tok(edit["nouvelle"], add_special_tokens=False)["input_ids"]
    entree = torch.tensor([ids_p + ids_o], device=model.device)
    labels = torch.tensor([[-100] * len(ids_p) + ids_o], device=model.device)
    for _ in range(p.pas_ft):
        perte = model(input_ids=entree, labels=labels).loss
        optim.zero_grad()
        perte.backward()
        optim.step()
        with torch.no_grad():
            mod.weight.copy_(W0 + (mod.weight - W0).clamp(-p.epsilon_ft, p.epsilon_ft))
    mod.weight.requires_grad_(False)
    return {"perte_finale": round(perte.item(), 4)}
