"""Extension de la fenêtre de contexte : modifier l'encodage positionnel RoPE puis fine-tuner brièvement sur des textes longs.

Méthodes (`--set contexte.methode=...`) : linear (interpolation de positions), dynamic (NTK dynamique),
yarn, abf (augmenter rope_theta, recette Llama 3 / Code Llama).
Étapes : mesurer le modèle d'origine -> mise à l'échelle seule -> fine-tuning long -> remesurer -> sauvegarder.

Mesures : clé cachée dans une phrase répétée (test facile), puis deux tâches plus dures sur du texte varié (paragraphes
Wikipédia jamais vus à l'entraînement) : une clé cachée (`cle_varie_L`) et plusieurs clés de même format dont il faut
retrouver celle d'un nom donné (`multi_L`, tâche multi-clés de RULER). Perplexité à 8 192 tokens et à 2 048 tokens (le contexte court ne doit pas se dégrader).
L'écart apparié entre le modèle entraîné et la mise à l'échelle seule dit ce qu'apporte l'entraînement.
"""

import pathlib
import random
import sys

sys.path.insert(0, str(next(p for p in pathlib.Path(__file__).resolve().parents if (p / "commun").is_dir())))

import torch
from transformers import AutoConfig, AutoModelForCausalLM

from commun import config, donnees, entrainement, evaluation, modeles, preuve, rapport

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


def remplissage_varie(tok, pool, longueur: int, rng) -> list[str]:
    """Paragraphes tirés au hasard, jusqu'à `longueur` tokens environ."""
    res, n = [], 0
    while n < longueur:
        p = rng.choice(pool)
        res.append(p)
        n += len(tok(p)["input_ids"])
    return res


def prompt_cle_varie(tok, pool, longueur: int, graine: int) -> tuple[str, str]:
    rng = random.Random(graine)
    cle = str(rng.randint(10000, 99999))
    para = remplissage_varie(tok, pool, longueur - 40, rng)
    para.insert(rng.randint(0, len(para)), f"The pass key is {cle}. Remember it. {cle} is the pass key.")
    return "\n".join(para) + "\nWhat is the pass key? The pass key is", cle


NOMS = ["Alder", "Birch", "Cedar", "Elm", "Hazel", "Maple", "Rowan", "Willow", "Aspen", "Juniper", "Laurel", "Spruce"]


def prompt_multi_cles(tok, pool, longueur: int, graine: int, n_cles: int) -> tuple[str, list[str]]:
    """Plusieurs clés de même format cachées dans le texte (tâche multi-clés de RULER) : il faut retrouver celle du nom
    demandé parmi des clés voisines, toutes présentes dans le contexte. Renvoie le prompt et les clés, la bonne en tête."""
    rng = random.Random(graine)
    noms = rng.sample(NOMS, n_cles)
    cles = [str(rng.randint(10000, 99999)) for _ in noms]
    para = remplissage_varie(tok, pool, longueur - 20 * n_cles - 30, rng)
    for nom, cle in zip(noms, cles):
        para.insert(rng.randint(0, len(para)), f"The pass key for {nom} is {cle}.")
    i = rng.randrange(n_cles)
    question = f"\nWhat is the pass key for {noms[i]}? The pass key for {noms[i]} is"
    return "\n".join(para) + question, [cles[i]] + cles[:i] + cles[i + 1 :]


@torch.no_grad()
def mesurer_dur(model, tok, cfg, pool, graine0: int) -> dict:
    """Sur du texte varié : une clé cachée (choix parmi 10 candidats), puis plusieurs clés de même format dont il faut
    retrouver la bonne (choix parmi les clés présentes dans le contexte)."""
    res = {}
    for L in cfg.evaluation.longueurs_dures:
        ok1, ok2 = [], []
        for g in range(cfg.evaluation.n_dur):
            prompt, cle = prompt_cle_varie(tok, pool, L, graine0 + g)
            rng = random.Random(5000 + graine0 + g)
            cands = [cle] + [str(rng.randint(10000, 99999)) for _ in range(9)]
            ok1.append(int(modeles.logprob_suites(model, tok, [prompt] * 10, [" " + c for c in cands]).argmax()) == 0)
            prompt, cands = prompt_multi_cles(tok, pool, L, graine0 + g, cfg.evaluation.n_cles_multi)
            ok2.append(int(modeles.logprob_suites(model, tok, [prompt] * len(cands), [" " + c for c in cands]).argmax()) == 0)
        res[f"cle_varie_{L}"] = preuve.taux(ok1, "taux")
        res[f"multi_{L}"] = preuve.taux(ok2, "taux")
    return res


@torch.no_grad()
def mesurer(model, tok, cfg, longs, pool=None):
    """Clé cachée en choix forcé : la vraie clé doit être la plus probable parmi 10 candidates.

    Mesure ce que le modèle a lu, indépendamment de sa façon de formuler (utile pour un modèle de base).
    """
    res = {}
    for L in cfg.evaluation.longueurs:
        ok = []
        for g in range(cfg.evaluation.n_cles):
            prompt, cle = prompt_cle(tok, L, g)
            rng = random.Random(1000 + g)
            candidates = [cle] + [str(rng.randint(10000, 99999)) for _ in range(9)]
            lp = modeles.logprob_suites(model, tok, [prompt] * len(candidates), [" " + c for c in candidates])
            ok.append(int(lp.argmax()) == 0)
        res[f"cle_{L}"] = preuve.taux(ok, "taux")
    res[f"perplexite_{cfg.evaluation.longueur_ppl}"] = evaluation.perplexite(model, tok, longs, cfg.evaluation.longueur_ppl, batch=1)
    res["perplexite_2048"] = evaluation.perplexite(model, tok, longs, 2048, batch=1)
    if pool:
        res.update(mesurer_dur(model, tok, cfg, pool, 100 * (cfg.graine - 42)))
    return res


def main():
    cfg = config.charger(__file__)
    c = cfg.contexte
    tok = modeles.charger_tokenizer(cfg.modele.nom)
    dtype = modeles.dtype_calcul() if cfg.modele.get("dtype", "auto") == "auto" else getattr(torch, cfg.modele.dtype)

    textes = donnees.textes_generaux(cfg.donnees.n_textes, graine=cfg.graine, min_caracteres=200)
    paragraphes_test = donnees.textes_generaux(cfg.evaluation.n_ppl * 60, graine=123, min_caracteres=200, split="test")
    longs = ["\n".join(paragraphes_test[i : i + 60]) for i in range(0, len(paragraphes_test), 60)]   # documents longs, jamais entraînés

    pool = [p for p in paragraphes_test if len(p) > 200]   # paragraphes de remplissage, jamais entraînés
    origine = AutoModelForCausalLM.from_pretrained(cfg.modele.nom, dtype=dtype).to(modeles.appareil())
    res = {"origine": mesurer(origine, tok, cfg, longs, pool)}
    del origine

    conf = configurer_rope(AutoConfig.from_pretrained(cfg.modele.nom), c)
    print(f"[contexte] {c.methode} : {conf.max_position_embeddings} positions, {getattr(conf, 'rope_parameters', None) or conf.rope_scaling}")
    model = AutoModelForCausalLM.from_pretrained(cfg.modele.nom, config=conf, dtype=dtype).to(modeles.appareil())
    res["echelle_seule"] = mesurer(model, tok, cfg, longs, pool)

    if cfg.entrainement.max_steps == 0:   # mise à l'échelle seule (ex. NTK dynamique), sans entraînement
        if cfg.get("sauvegarder", True):
            modeles.sauvegarder(model, tok, f"{cfg.sortie}/modele-{c.methode}")
        rapport.enregistrer(cfg, **res)
        return

    trainer = entrainement.lm_causal(model, tok, textes, cfg.entrainement, cfg.sortie, cfg.entrainement.longueur)
    res["apres_finetuning"] = mesurer(trainer.model, tok, cfg, longs, pool)
    # Ce qu'apporte l'entraînement, à mise à l'échelle égale : écart apparié sur chaque tâche.
    res["apres_moins_echelle"] = {k: preuve.difference_appariee(res["echelle_seule"][k]["succes"], res["apres_finetuning"][k]["succes"])
                                  for k in res["apres_finetuning"] if isinstance(res["apres_finetuning"][k], dict) and "succes" in res["apres_finetuning"][k]}

    if cfg.get("sauvegarder", True):
        trainer.save_model(f"{cfg.sortie}/modele-{c.methode}")
        tok.save_pretrained(f"{cfg.sortie}/modele-{c.methode}")
    rapport.enregistrer(cfg, **res)


if __name__ == "__main__":
    main()
