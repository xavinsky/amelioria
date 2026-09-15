"""Faire grandir un modèle existant au lieu d'en pré-entraîner un plus gros de zéro.

- profondeur        : depth up-scaling (SOLAR) — dupliquer un bloc de couches du milieu.
- expansion_blocs   : block expansion (LLaMA Pro) — insérer des couches initialisées à l'identité, seules entraînées.
- moe               : sparse upcycling — chaque MLP dense devient E experts identiques + un routeur (dense -> Mixtral).
Écrit pour la famille LLaMA / Mistral / SmolLM (`model.model.layers`, `self_attn.o_proj`, `mlp.{gate,up,down}_proj`).
"""

from __future__ import annotations

import copy

import torch
from transformers import MixtralConfig, MixtralForCausalLM


def _renumeroter(model):
    for i, couche in enumerate(model.model.layers):
        if hasattr(couche.self_attn, "layer_idx"):
            couche.self_attn.layer_idx = i   # indice utilisé par le cache KV
        if hasattr(couche, "layer_idx"):
            couche.layer_idx = i
    model.config.num_hidden_layers = len(model.model.layers)
    if getattr(model.config, "layer_types", None):
        model.config.layer_types = [model.config.layer_types[0]] * len(model.model.layers)


def profondeur(model, m: int):
    """SOLAR : n couches -> couches [0, n-m) + [m, n) = 2(n-m) couches. Les m couches centrales sont « cousues »."""
    couches = model.model.layers
    n = len(couches)
    assert 0 < m < n, "m doit être compris entre 1 et n-1"
    nouvelles = [couches[i] for i in range(n - m)] + [copy.deepcopy(couches[i]) for i in range(m, n)]
    model.model.layers = torch.nn.ModuleList(nouvelles)
    _renumeroter(model)
    return model


def expansion_blocs(model, tous_les: int):
    """LLaMA Pro : après chaque groupe de `tous_les` couches, une copie dont les projections de sortie sont à zéro.

    La nouvelle couche renvoie donc exactement son entrée (connexion résiduelle) : le modèle agrandi est
    identique à l'original avant entraînement. Renvoie aussi les indices des nouvelles couches.
    """
    nouvelles, indices = [], []
    for i, couche in enumerate(model.model.layers):
        nouvelles.append(couche)
        if (i + 1) % tous_les == 0:
            bloc = copy.deepcopy(couche)
            torch.nn.init.zeros_(bloc.self_attn.o_proj.weight)
            torch.nn.init.zeros_(bloc.mlp.down_proj.weight)
            indices.append(len(nouvelles))
            nouvelles.append(bloc)
    model.model.layers = torch.nn.ModuleList(nouvelles)
    _renumeroter(model)
    return model, indices


@torch.no_grad()
def moe(model, n_experts: int, top_k: int, bruit: float = 0.0, aux: float = 0.01):
    """Dense LLaMA/Mistral -> Mixtral : attention, normes et embeddings copiés, MLP copié dans chaque expert."""
    c = model.config
    assert not getattr(c, "attention_bias", False) and not getattr(c, "mlp_bias", False), "Mixtral n'a pas de biais : famille LLaMA/Mistral uniquement"
    conf = MixtralConfig(
        vocab_size=c.vocab_size, hidden_size=c.hidden_size, intermediate_size=c.intermediate_size,
        num_hidden_layers=c.num_hidden_layers, num_attention_heads=c.num_attention_heads,
        num_key_value_heads=c.num_key_value_heads, head_dim=getattr(c, "head_dim", None), hidden_act=c.hidden_act,
        max_position_embeddings=c.max_position_embeddings, rms_norm_eps=c.rms_norm_eps,
        tie_word_embeddings=c.tie_word_embeddings, rope_parameters=getattr(c, "rope_parameters", None),
        bos_token_id=c.bos_token_id, eos_token_id=c.eos_token_id, pad_token_id=c.pad_token_id,
        num_local_experts=n_experts, num_experts_per_tok=top_k, router_aux_loss_coef=aux, output_router_logits=True,
    )
    moe_model = MixtralForCausalLM(conf).to(dtype=model.dtype)
    etat = model.state_dict()
    cible = moe_model.state_dict()
    for nom in cible:
        if nom in etat and etat[nom].shape == cible[nom].shape:
            cible[nom].copy_(etat[nom])
    for i in range(c.num_hidden_layers):
        mlp = model.model.layers[i].mlp
        experts = moe_model.model.layers[i].mlp.experts
        gate_up = torch.cat([mlp.gate_proj.weight, mlp.up_proj.weight], dim=0)
        for e in range(n_experts):
            experts.gate_up_proj[e].copy_(gate_up + bruit * gate_up.std() * torch.randn_like(gate_up))
            experts.down_proj[e].copy_(mlp.down_proj.weight)
        torch.nn.init.normal_(moe_model.model.layers[i].mlp.gate.weight, std=0.02)
    if c.tie_word_embeddings:
        moe_model.tie_weights()
    return moe_model
