"""Croissance d'architecture : agrandir un modèle pré-entraîné (profondeur, blocs identité, MoE) puis le « cicatriser ».

Étapes : mesurer le modèle d'origine -> transformer l'architecture -> mesurer (effet immédiat) ->
pré-entraînement continu sur un nouveau domaine -> remesurer -> sauvegarder.

Besoin visé (LLaMA Pro) : apprendre un domaine nouveau, ici une langue, sans perdre ce que le modèle savait. Deux
perplexités sur des textes jamais vus : celle du domaine (ce qu'on apprend) et celle de l'anglais général (ce
qu'on ne doit pas perdre). Témoin : le modèle d'origine entraîné en entier sur les mêmes textes, même nombre de pas.
"""

import pathlib
import sys

sys.path.insert(0, str(next(p for p in pathlib.Path(__file__).resolve().parents if (p / "commun").is_dir())))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import torch  # noqa: E402
from transformers import TrainerCallback  # noqa: E402

import croissance  # noqa: E402
from commun import config, donnees, entrainement, evaluation, modeles, rapport  # noqa: E402


def textes_domaine(d, graine):
    """Articles du domaine (jeu Hugging Face), séparés en entraînement et test (un sur cinq, jamais vu)."""
    from datasets import load_dataset

    ds = load_dataset(d.nom, d.get("config"), split="train").shuffle(seed=graine)
    docs = [t for t in ds.select(range(d.n_textes))[d.colonne] if t.strip()]
    return [t for i, t in enumerate(docs) if i % 5 != 4], [t for i, t in enumerate(docs) if i % 5 == 4][: d.n_test]


class Courbe(TrainerCallback):
    """Perplexités du domaine et de l'anglais général tous les `tous_les` pas, dans les conditions de
    l'entraînement : les courbes des deux modèles se comparent à apprentissage du domaine égal."""

    def __init__(self, tok, test, test_domaine, tous_les):
        self.tok, self.test, self.test_domaine, self.tous_les, self.points = tok, test, test_domaine, tous_les, []

    def on_step_end(self, args, state, control, model=None, **kw):
        if state.global_step % self.tous_les == 0:
            self.points.append({"pas": state.global_step,
                                "perplexite_domaine": evaluation.perplexite(model, self.tok, self.test_domaine, 256),
                                "perplexite_generale": evaluation.perplexite(model, self.tok, self.test, 256)})
            model.train()


def a_domaine_egal(courbe_methode, courbe_temoin):
    """Premier relevé du témoin qui atteint le niveau final de la méthode dans le domaine (perplexité du domaine
    inférieure ou égale) : on compare alors ce que chacun a perdu en anglais général."""
    cible = courbe_methode[-1]["perplexite_domaine"]
    atteint = [p for p in courbe_temoin if p["perplexite_domaine"] <= cible]
    if not atteint:
        return {}
    t = atteint[0]
    return {"perplexite_domaine_visee": cible, "pas_du_temoin": t["pas"],
            "temoin_perplexite_domaine": t["perplexite_domaine"],
            "temoin_perplexite_generale": t["perplexite_generale"],
            "methode_perplexite_generale": courbe_methode[-1]["perplexite_generale"]}


def mesurer(model, tok, test, reference_logits=None, sonde=None, test_domaine=None):
    modeles.sans_autocast(model)
    res = {"parametres_M": round(sum(p.numel() for p in model.parameters()) / 1e6, 1),
           "parametres_entraines_M": round(sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6, 1),
           "couches": len(model.model.layers),
           **({"perplexite_domaine": evaluation.perplexite(model, tok, test_domaine, 256)} if test_domaine else {}),
           "perplexite_generale": evaluation.perplexite(model, tok, test, 256)}
    if reference_logits is not None:
        with torch.no_grad():
            res["ecart_max_logits_vs_origine"] = round((model(sonde).logits.float() - reference_logits).abs().max().item(), 5)
    return res


def main():
    cfg = config.charger(__file__)
    c = cfg.croissance
    model, tok = modeles.charger_llm(cfg.modele.nom, cfg.modele.get("dtype", "auto"))
    test = donnees.textes_generaux(cfg.donnees.n_test, graine=123, split="test")
    if cfg.get("domaine"):
        train, test_domaine = textes_domaine(cfg.domaine, cfg.graine)
    else:
        train, test_domaine = donnees.textes_generaux(cfg.donnees.n_textes, graine=cfg.graine), None
    sonde = tok(test[0], return_tensors="pt", truncation=True, max_length=64).input_ids.to(model.device)
    with torch.no_grad():
        ref = model(sonde).logits.float()

    res = {"origine": mesurer(model, tok, test, test_domaine=test_domaine)}
    a_entrainer = None
    if c.methode == "profondeur":
        model = croissance.profondeur(model, c.m)
    elif c.methode == "expansion_blocs":
        model, a_entrainer = croissance.expansion_blocs(model, c.tous_les)
    elif c.methode == "moe":
        model = croissance.moe(model, c.n_experts, c.top_k, c.bruit_experts).to(modeles.appareil())
    else:
        raise ValueError(c.methode)
    res["transforme"] = mesurer(model, tok, test, ref, sonde, test_domaine)
    print(f"[croissance] {c.methode} : {res['origine']['parametres_M']}M -> {res['transforme']['parametres_M']}M paramètres")

    if a_entrainer is not None:   # LLaMA Pro : seules les couches ajoutées apprennent, l'original est préservé
        for i, couche in enumerate(model.model.layers):
            couche.requires_grad_(i in a_entrainer)
        model.model.embed_tokens.requires_grad_(False)
        model.lm_head.requires_grad_(False)

    suivi = cfg.get("suivi_tous_les") if test_domaine else None
    courbe = Courbe(tok, test, test_domaine, suivi) if suivi else None
    trainer = entrainement.lm_causal(model, tok, train, cfg.entrainement, cfg.sortie, cfg.entrainement.longueur_bloc,
                                     callbacks=[courbe] if courbe else None)
    res["apres_entrainement"] = mesurer(trainer.model, tok, test, test_domaine=test_domaine)

    if cfg.get("sauvegarder", True):
        trainer.save_model(f"{cfg.sortie}/modele-{c.methode}")
        tok.save_pretrained(f"{cfg.sortie}/modele-{c.methode}")

    if cfg.get("temoin", True):
        # Témoin : le modèle d'origine, sans croissance, entraîné en entier sur les mêmes textes pendant le même
        # nombre de pas. Il montre ce que fait le simple pré-entraînement continu : apprentissage du domaine et oubli.
        entrainement.liberer(trainer)
        del trainer, model
        temoin, _ = modeles.charger_llm(cfg.modele.nom, cfg.modele.get("dtype", "auto"))
        courbe_temoin = Courbe(tok, test, test_domaine, suivi) if suivi else None
        trainer = entrainement.lm_causal(temoin, tok, train, cfg.entrainement, f"{cfg.sortie}/temoin",
                                         cfg.entrainement.longueur_bloc, callbacks=[courbe_temoin] if courbe_temoin else None)
        res["temoin"] = mesurer(trainer.model, tok, test, test_domaine=test_domaine)
        if courbe and courbe.points:
            res["a_domaine_egal"] = a_domaine_egal(courbe.points, courbe_temoin.points)
            res["courbes"] = {"methode": courbe.points, "temoin": courbe_temoin.points}
    rapport.enregistrer(cfg, **res)


if __name__ == "__main__":
    main()
