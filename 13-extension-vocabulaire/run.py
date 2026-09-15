"""Extension de vocabulaire : ajouter des tokens pour un domaine ou une langue, les initialiser intelligemment, les entraîner.

Étapes : choisir les nouveaux tokens (fréquents et mal découpés) -> agrandir les embeddings (init = moyenne des
sous-tokens) -> phase 1 : n'entraîner que les embeddings -> phase 2 : entraînement complet -> mesurer.
Mesures comparables entre tokenizers : nombre de tokens du corpus (fertilité) et bits par caractère.
"""

import collections
import math
import pathlib
import re
import sys

sys.path.insert(0, str(next(p for p in pathlib.Path(__file__).resolve().parents if (p / "commun").is_dir())))

import torch

from commun import config, donnees, entrainement, modeles, rapport


def candidats(tok, textes, n, min_freq):
    """Mots fréquents découpés en plusieurs tokens, classés par nombre de tokens économisés."""
    freq = collections.Counter(m for t in textes for m in re.findall(r"[A-Za-zÀ-ÿ][\w'-]+", t))
    gains = []
    for mot, f in freq.items():
        if f < min_freq:
            continue
        pieces = len(tok(" " + mot, add_special_tokens=False)["input_ids"])
        if pieces > 1:
            gains.append((f * (pieces - 1), mot))
    return [m for _, m in sorted(gains, reverse=True)[:n]]


@torch.no_grad()
def bits_par_caractere(model, tok, textes):
    dev = next(model.parameters()).device
    nll, car = 0.0, 0
    for t in textes:
        ids = tok(t, return_tensors="pt").input_ids.to(dev)
        nll += model(ids, labels=ids).loss.item() * (ids.shape[1] - 1)
        car += len(t)
    return round(nll / car / math.log(2), 4)


@torch.no_grad()
def agrandir(model, tok_ancien, tok, nouveaux, init):
    n_ancien = model.get_input_embeddings().weight.shape[0]
    model.resize_token_embeddings(len(tok), mean_resizing=False)
    entree, sortie = model.get_input_embeddings().weight, model.get_output_embeddings().weight
    liees = entree.data_ptr() == sortie.data_ptr()
    for mot in nouveaux:
        i = tok.convert_tokens_to_ids(mot)
        if i < n_ancien:
            continue
        anciens = tok_ancien(" " + mot, add_special_tokens=False)["input_ids"]
        if init == "moyenne":
            entree[i] = entree[anciens].mean(0)
            if not liees:
                sortie[i] = sortie[anciens].mean(0)
        else:
            entree[i].normal_(0, entree[:n_ancien].std().item())
            if not liees:
                sortie[i].normal_(0, sortie[:n_ancien].std().item())


def main():
    cfg = config.charger(__file__)
    model, tok = modeles.charger_llm(cfg.modele.nom, cfg.modele.get("dtype", "auto"))
    tok_ancien = modeles.charger_tokenizer(cfg.modele.nom)

    if cfg.donnees.source == "faits":
        kb = donnees.faits_fictifs(cfg.donnees.n_entites, cfg.donnees.graine)
        docs = [d["texte"] for d in kb["documents"]]
    else:
        from datasets import load_dataset

        ds = load_dataset(cfg.donnees.nom, cfg.donnees.get("config"), split="train").shuffle(seed=cfg.graine)
        docs = [t for t in ds.select(range(cfg.donnees.n_textes))[cfg.donnees.colonne] if t.strip()]
    train = [t for i, t in enumerate(docs) if i % 5 != 4]
    test = [t for i, t in enumerate(docs) if i % 5 == 4]

    nouveaux = cfg.vocabulaire.get("liste") or candidats(tok, train, cfg.vocabulaire.n_nouveaux, cfg.vocabulaire.min_freq)
    print(f"[vocabulaire] {len(nouveaux)} nouveaux tokens, ex. {nouveaux[:8]}")
    mesure = lambda m, t: {"tokens_corpus_test": sum(len(t(x)["input_ids"]) for x in test), "bits_par_caractere": bits_par_caractere(m, t, test)}
    avant = mesure(model, tok_ancien)

    tok.add_tokens(nouveaux)
    agrandir(model, tok_ancien, tok, nouveaux, cfg.vocabulaire.init)
    apres_init = mesure(model, tok)

    # Phase 1 : seuls les embeddings (entrée et sortie) apprennent, le reste du modèle est gelé.
    for nom, p in model.named_parameters():
        p.requires_grad_("embed" in nom or "lm_head" in nom or "wte" in nom)
    entrainement.lm_causal(model, tok, train, cfg.phase1, f"{cfg.sortie}/phase1", cfg.donnees.longueur_bloc)
    apres_phase1 = mesure(model, tok)

    # Phase 2 : tout le modèle (ou LoRA + embeddings, voir README), lr plus faible.
    for p in model.parameters():
        p.requires_grad_(True)
    trainer = entrainement.lm_causal(model, tok, train, cfg.phase2, f"{cfg.sortie}/phase2", cfg.donnees.longueur_bloc)
    apres_phase2 = mesure(trainer.model, tok)

    if cfg.get("sauvegarder", True):
        trainer.save_model(f"{cfg.sortie}/modele")
        tok.save_pretrained(f"{cfg.sortie}/modele")
    rapport.enregistrer(cfg, avant=avant, apres_init=apres_init, apres_phase1=apres_phase1, apres_phase2=apres_phase2,
                        nouveaux_tokens=nouveaux)


if __name__ == "__main__":
    main()
