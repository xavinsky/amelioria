"""Algorithmes de fusion de poids, en PyTorch pur, pour n'importe quels modèles de même architecture.

Chaque fonction reçoit des tenseurs de même forme (un par modèle) et renvoie le tenseur fusionné.
Notation : base = modèle de départ commun, taches[i] = θ_i - base (« vecteur de tâche »).
"""

from __future__ import annotations

import torch


def lineaire(tenseurs: list[torch.Tensor], poids: list[float]) -> torch.Tensor:
    """Moyenne pondérée (model soup). Ne demande pas de modèle de base."""
    total = sum(poids)
    return sum(w / total * t for w, t in zip(poids, tenseurs))


def slerp(a: torch.Tensor, b: torch.Tensor, t: float, eps: float = 1e-8) -> torch.Tensor:
    """Interpolation sphérique entre deux modèles : préserve la norme mieux qu'une moyenne linéaire."""
    a_n, b_n = a / (a.norm() + eps), b / (b.norm() + eps)
    cos = torch.clamp((a_n * b_n).sum(), -1.0, 1.0)
    omega = torch.arccos(cos)
    if omega.abs() < 1e-4:
        return (1 - t) * a + t * b
    s = torch.sin(omega)
    return torch.sin((1 - t) * omega) / s * a + torch.sin(t * omega) / s * b


def arithmetique_taches(base: torch.Tensor, tenseurs: list[torch.Tensor], poids: list[float]) -> torch.Tensor:
    """Task arithmetic : base + Σ λ_i (θ_i - base)."""
    return base + sum(w * (t - base) for w, t in zip(poids, tenseurs))


def _garder_top(delta: torch.Tensor, densite: float) -> torch.Tensor:
    if densite >= 1.0:
        return delta
    k = max(int(delta.numel() * densite), 1)
    seuil = delta.abs().flatten().float().kthvalue(delta.numel() - k + 1).values
    return delta * (delta.abs() >= seuil)


def _abandon_aleatoire(delta: torch.Tensor, densite: float, generateur: torch.Generator) -> torch.Tensor:
    """DARE : met à zéro (1 - densite) des coordonnées au hasard et rééchelonne le reste par 1/densite."""
    if densite >= 1.0:
        return delta
    masque = torch.rand(delta.shape, generator=generateur) < densite
    return delta * masque / densite


def ties(base, tenseurs, poids, densite: float = 0.2, abandon: str = "magnitude", generateur=None):
    """TIES-merging (+ DARE si abandon='aleatoire') :
    1) élaguer chaque vecteur de tâche (top-k% magnitude, ou aléatoire pour DARE),
    2) élire un signe par coordonnée (signe de la somme pondérée),
    3) moyenner uniquement les valeurs d'accord avec le signe élu.
    """
    deltas = []
    for w, t in zip(poids, tenseurs):
        d = t.float() - base.float()
        d = _garder_top(d, densite) if abandon == "magnitude" else _abandon_aleatoire(d, densite, generateur)
        deltas.append(w * d)
    pile = torch.stack(deltas)
    signe = torch.sign(pile.sum(0))
    accord = (torch.sign(pile) == signe) & (pile != 0)
    somme = (pile * accord).sum(0)
    compte = accord.sum(0).clamp(min=1)
    return (base.float() + somme / compte).to(base.dtype)


def dare_lineaire(base, tenseurs, poids, densite: float = 0.5, generateur=None):
    """DARE sans élection de signe : abandon aléatoire puis arithmétique des tâches."""
    return (base.float() + sum(w * _abandon_aleatoire(t.float() - base.float(), densite, generateur)
                               for w, t in zip(poids, tenseurs))).to(base.dtype)


@torch.no_grad()
def ecart_relatif(etat: dict, etat_base: dict) -> float:
    """‖θ - θ_base‖ / ‖θ_base‖ sur les tenseurs communs : vérifie qu'un modèle est bien un fine-tune de la base.

    Un fine-tune s'éloigne de quelques pour cent au plus. Un modèle issu d'un pré-entraînement continu massif
    (Qwen2.5-Math, Qwen2.5-Coder) est à plus de 50 % : ses vecteurs de tâche ne se fusionnent pas.
    """
    num = den = 0.0
    for nom, t in etat_base.items():
        if nom in etat and etat[nom].shape == t.shape and t.is_floating_point():
            num += (etat[nom].float() - t.float()).pow(2).sum().item()
            den += t.float().pow(2).sum().item()
    return round((num / max(den, 1e-12)) ** 0.5, 4)


@torch.no_grad()
def fusionner_etats(methode: str, etat_base: dict | None, etats: list[dict], poids: list[float], densite: float = 0.5,
                    t_slerp: float = 0.5, graine: int = 0) -> dict:
    """Applique `methode` tenseur par tenseur. Les tenseurs de formes différentes (vocabulaire étendu...) viennent du 1er modèle."""
    g = torch.Generator().manual_seed(graine)
    reference = etat_base if etat_base is not None else etats[0]
    fusion = {}
    for nom, t0 in reference.items():
        tenseurs = [e[nom] for e in etats if nom in e and e[nom].shape == t0.shape]
        if len(tenseurs) != len(etats) or not t0.is_floating_point():
            fusion[nom] = etats[0].get(nom, t0)
            continue
        if methode == "lineaire":
            fusion[nom] = lineaire(tenseurs, poids)
        elif methode == "slerp":
            assert len(tenseurs) == 2, "slerp fusionne exactement deux modèles"
            fusion[nom] = slerp(tenseurs[0].float(), tenseurs[1].float(), t_slerp).to(t0.dtype)
        elif methode == "arithmetique_taches":
            fusion[nom] = arithmetique_taches(t0, tenseurs, poids)
        elif methode == "ties":
            fusion[nom] = ties(t0, tenseurs, poids, densite, "magnitude", g)
        elif methode == "dare_ties":
            fusion[nom] = ties(t0, tenseurs, poids, densite, "aleatoire", g)
        elif methode == "dare_lineaire":
            fusion[nom] = dare_lineaire(t0, tenseurs, poids, densite, g)
        else:
            raise ValueError(f"méthode de fusion inconnue : {methode}")
    return fusion
