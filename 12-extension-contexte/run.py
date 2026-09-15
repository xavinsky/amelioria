"""Extension de la fenêtre de contexte : modifier l'encodage positionnel RoPE puis fine-tuner brièvement sur des textes longs.

Méthodes (`--set contexte.methode=...`) : linear (interpolation de positions), dynamic (NTK dynamique),
yarn, abf (augmenter rope_theta, recette Llama 3 / Code Llama).
Étapes : mesurer le modèle d'origine -> mise à l'échelle seule -> fine-tuning long -> remesurer -> sauvegarder.
"""

import pathlib
import random
import sys

sys.path.insert(0, str(next(p for p in pathlib.Path(__file__).resolve().parents if (p / "commun").is_dir())))

import torch
from transformers import AutoConfig, AutoModelForCausalLM

from commun import config, donnees, entrainement, evaluation, modeles, rapport

REMPLISSAGE = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. "


def configurer_rope(conf, c):
    """Écrit la mise à l'échelle RoPE dans la config (format transformers >= 5 `rope_parameters`, ou ancien `rope_scaling`)."""
    origine = c.get("original") or conf.max_position_embeddings
    params = dict(getattr(conf, "rope_parameters", None) or {})
    theta = params.get("rope_theta", getattr(conf, "rope_theta", 10000.0))
    if c.methode == "abf":
        nouveau = {"rope_type": "default", "rope_theta": theta * c.facteur_theta}
    else:
        nouveau = {"rope_type": c.methode, "rope_theta": theta, "factor": float(c.facteur)}
        if c.methode in ("yarn", "longrope", "llama3"):
            nouveau["original_max_position_embeddings"] = origine
    if hasattr(conf, "rope_parameters"):
        conf.rope_parameters = nouveau
    else:
        conf.rope_scaling = {k: v for k, v in nouveau.items() if k != "rope_theta"}
        conf.rope_theta = nouveau["rope_theta"]
    conf.max_position_embeddings = int(origine * c.facteur)
    return conf


def prompt_cle(tok, longueur: int, graine: int) -> tuple[str, str]:
    """Test « passkey » : une clé cachée au milieu d'un texte de remplissage de `longueur` tokens."""
    rng = random.Random(graine)
    cle = str(rng.randint(10000, 99999))
    n_rep = max(longueur // len(tok(REMPLISSAGE)["input_ids"]), 1)
    position = rng.randint(0, n_rep)
    texte = REMPLISSAGE * position + f"The pass key is {cle}. Remember it. {cle} is the pass key. " + REMPLISSAGE * (n_rep - position)
    return texte + "\nWhat is the pass key? The pass key is", cle


@torch.no_grad()
def mesurer(model, tok, cfg, longs):
    res = {}
    for L in cfg.evaluation.longueurs:
        ok = 0
        for g in range(cfg.evaluation.n_cles):
            prompt, cle = prompt_cle(tok, L, g)
            ok += cle in modeles.generer(model, tok, [prompt], max_new_tokens=8, batch=1)[0]
        res[f"cle_{L}"] = round(ok / cfg.evaluation.n_cles, 3)
    res[f"perplexite_{cfg.evaluation.longueur_ppl}"] = evaluation.perplexite(model, tok, longs, cfg.evaluation.longueur_ppl, batch=1)
    return res


def main():
    cfg = config.charger(__file__)
    c = cfg.contexte
    tok = modeles.charger_tokenizer(cfg.modele.nom)
    dtype = modeles.dtype_calcul() if cfg.modele.get("dtype", "auto") == "auto" else getattr(torch, cfg.modele.dtype)

    textes = donnees.textes_generaux(cfg.donnees.n_textes, graine=cfg.graine, min_caracteres=200)
    longs = ["\n".join(textes[i : i + 60]) for i in range(0, cfg.evaluation.n_ppl * 60, 60)]   # documents longs pour la perplexité

    origine = AutoModelForCausalLM.from_pretrained(cfg.modele.nom, dtype=dtype).to(modeles.appareil())
    res = {"origine": mesurer(origine, tok, cfg, longs)}
    del origine

    conf = configurer_rope(AutoConfig.from_pretrained(cfg.modele.nom), c)
    print(f"[contexte] {c.methode} : {conf.max_position_embeddings} positions, {getattr(conf, 'rope_parameters', None) or conf.rope_scaling}")
    model = AutoModelForCausalLM.from_pretrained(cfg.modele.nom, config=conf, dtype=dtype).to(modeles.appareil())
    res["echelle_seule"] = mesurer(model, tok, cfg, longs)

    if cfg.entrainement.max_steps == 0:   # mise à l'échelle seule (ex. NTK dynamique), sans entraînement
        if cfg.get("sauvegarder", True):
            modeles.sauvegarder(model, tok, f"{cfg.sortie}/modele-{c.methode}")
        rapport.enregistrer(cfg, **res)
        return

    trainer = entrainement.lm_causal(model, tok, textes, cfg.entrainement, cfg.sortie, cfg.entrainement.longueur)
    res["apres_finetuning"] = mesurer(trainer.model, tok, cfg, longs)

    if cfg.get("sauvegarder", True):
        trainer.save_model(f"{cfg.sortie}/modele-{c.methode}")
        tok.save_pretrained(f"{cfg.sortie}/modele-{c.methode}")
    rapport.enregistrer(cfg, **res)


if __name__ == "__main__":
    main()
