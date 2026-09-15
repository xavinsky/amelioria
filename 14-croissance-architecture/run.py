"""Croissance d'architecture : agrandir un modèle pré-entraîné (profondeur, blocs identité, MoE) puis le « cicatriser ».

Étapes : mesurer le modèle d'origine -> transformer l'architecture -> mesurer (effet immédiat) ->
pré-entraînement continu de cicatrisation -> remesurer -> sauvegarder.
"""

import pathlib
import sys

sys.path.insert(0, str(next(p for p in pathlib.Path(__file__).resolve().parents if (p / "commun").is_dir())))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import torch  # noqa: E402

import croissance  # noqa: E402
from commun import config, donnees, entrainement, evaluation, modeles, rapport  # noqa: E402


def mesurer(model, tok, test, reference_logits=None, sonde=None):
    res = {"parametres_M": round(sum(p.numel() for p in model.parameters()) / 1e6, 1),
           "couches": len(model.model.layers),
           "perplexite_generale": evaluation.perplexite(model, tok, test, 256)}
    if reference_logits is not None:
        with torch.no_grad():
            res["ecart_max_logits_vs_origine"] = round((model(sonde).logits.float() - reference_logits).abs().max().item(), 5)
    return res


def main():
    cfg = config.charger(__file__)
    c = cfg.croissance
    model, tok = modeles.charger_llm(cfg.modele.nom, cfg.modele.get("dtype", "auto"))
    textes = donnees.textes_generaux(cfg.donnees.n_textes + cfg.donnees.n_test, graine=cfg.graine)
    train, test = textes[cfg.donnees.n_test :], textes[: cfg.donnees.n_test]
    sonde = tok(test[0], return_tensors="pt", truncation=True, max_length=64).input_ids.to(model.device)
    with torch.no_grad():
        ref = model(sonde).logits.float()

    res = {"origine": mesurer(model, tok, test)}
    a_entrainer = None
    if c.methode == "profondeur":
        model = croissance.profondeur(model, c.m)
    elif c.methode == "expansion_blocs":
        model, a_entrainer = croissance.expansion_blocs(model, c.tous_les)
    elif c.methode == "moe":
        model = croissance.moe(model, c.n_experts, c.top_k, c.bruit_experts).to(modeles.appareil())
    else:
        raise ValueError(c.methode)
    res["transforme"] = mesurer(model, tok, test, ref, sonde)
    print(f"[croissance] {c.methode} : {res['origine']['parametres_M']}M -> {res['transforme']['parametres_M']}M paramètres")

    if a_entrainer is not None:   # LLaMA Pro : seules les couches ajoutées apprennent, l'original est préservé
        for i, couche in enumerate(model.model.layers):
            couche.requires_grad_(i in a_entrainer)
        model.model.embed_tokens.requires_grad_(False)
        model.lm_head.requires_grad_(False)

    trainer = entrainement.lm_causal(model, tok, train, cfg.entrainement, cfg.sortie, cfg.entrainement.longueur_bloc)
    res["apres_cicatrisation"] = mesurer(trainer.model, tok, test)

    if cfg.get("sauvegarder", True):
        trainer.save_model(f"{cfg.sortie}/modele-{c.methode}")
        tok.save_pretrained(f"{cfg.sortie}/modele-{c.methode}")
    rapport.enregistrer(cfg, **res)


if __name__ == "__main__":
    main()
