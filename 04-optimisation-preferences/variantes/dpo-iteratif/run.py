"""DPO itératif « on-policy » : le modèle génère lui-même les paires à chaque tour.

À chaque tour : k réponses échantillonnées par demande -> un modèle de récompense (`classement`) les note ->
la mieux notée = chosen, la moins bien notée = rejected -> un tour de DPO LoRA, fusionné dans le modèle, qui
devient la référence du tour suivant. C'est la recette d'Iterative DPO / Online DPO, sans préférences humaines.

Mesure : réponses gloutonnes aux mêmes demandes de test après chaque tour, notées par un autre modèle de
récompense (`juge`), plus grand que celui qui a fabriqué les paires ; taux de victoire contre le modèle de départ.
"""

import gc
import json
import pathlib
import sys

RACINE = next(p for p in pathlib.Path(__file__).resolve().parents if (p / "commun").is_dir())
sys.path.insert(0, str(RACINE))
sys.path.insert(0, str(RACINE / "04-optimisation-preferences"))

import preferences  # noqa: E402
import torch  # noqa: E402

from commun import config, entrainement, evaluation, modeles, preuve, rapport, reprise  # noqa: E402


def prompts_entrainement(cfg, tour: int, longueur_max: int = 2000) -> list[str]:
    """Demandes neuves à chaque tour, tirées du split d'entraînement du jeu de préférences (réponses ignorées)."""
    from datasets import load_dataset

    d, n = cfg.donnees, cfg.donnees.n_prompts_par_tour
    ds = load_dataset(d.nom, split=d.split_train).shuffle(seed=d.graine)
    prompts = [ex["chosen"][0]["content"] for ex in ds.select(range(min(len(ds), 3 * n * cfg.tours)))
               if len(ex["chosen"][0]["content"]) <= longueur_max]
    return prompts[(tour - 1) * n : tour * n]


def generer_paires(cfg, model, tok, prompts: list[str]) -> tuple[list[dict], dict]:
    g = cfg.generation
    textes = [modeles.prompt_texte(tok, p) for p in prompts]
    echantillons = modeles.generer(model, tok, textes, max_new_tokens=g.max_new_tokens,
                                   temperature=g.temperature, n=g.k, batch=g.batch)
    a_plat = [r.strip() for rs in echantillons for r in rs]
    notes = preferences.noter(cfg.classement, [p for p in prompts for _ in range(g.k)], {"e": a_plat})["e"]
    paires, ecarts = [], []
    for i, p in enumerate(prompts):
        n = notes[i * g.k : (i + 1) * g.k]
        r = a_plat[i * g.k : (i + 1) * g.k]
        haut, bas = max(range(g.k), key=n.__getitem__), min(range(g.k), key=n.__getitem__)
        if n[haut] - n[bas] >= cfg.get("ecart_min", 0.0) and r[haut] != r[bas]:
            paires.append(preferences.paire(tok, p, r[haut], r[bas]))
            ecarts.append(n[haut] - n[bas])
    stats = {"paires": len(paires), "ecart_moyen": round(sum(ecarts) / max(len(ecarts), 1), 3),
             "note_moyenne_echantillons": round(sum(notes) / len(notes), 3)}
    return paires, stats


def main():
    cfg = config.charger(__file__)
    model, tok = modeles.charger_llm(cfg.modele.nom, cfg.modele.get("dtype", "auto"))
    test = preferences.prompts_test(cfg, cfg.evaluation.n_prompts)

    def mesurer(m):
        return {"mesures": evaluation.evaluer(m, tok, cfg), "reponses": preferences.repondre(m, tok, test, cfg.evaluation)}

    etapes, reponses = {}, {}
    r = reprise.etape("tour_0", lambda: mesurer(model))
    etapes["tour_0"], reponses["tour_0"] = r["mesures"], r["reponses"]

    for tour in range(1, cfg.tours + 1):
        def un_tour():
            paires, stats = generer_paires(cfg, model, tok, prompts_entrainement(cfg, tour))
            print(f"[tour {tour}] {stats}", flush=True)
            trainer = preferences.construire_trainer(cfg, model, tok, paires, f"{cfg.sortie}/tour-{tour}")
            with entrainement.mesurer() as cout:
                reprise.entrainer(trainer)
            entrainement.liberer(trainer)
            fusionne = trainer.model.merge_and_unload() if hasattr(trainer.model, "merge_and_unload") else trainer.model
            fusionne.save_pretrained(f"{cfg.sortie}/tour-{tour}/modele")
            tok.save_pretrained(f"{cfg.sortie}/tour-{tour}/modele")
            res = mesurer(fusionne)
            res["mesures"] |= stats | cout
            return res

        r = reprise.etape(f"tour_{tour}", un_tour)
        etapes[f"tour_{tour}"], reponses[f"tour_{tour}"] = r["mesures"], r["reponses"]
        # Le modèle du tour (fusionné) sert de point de départ et de référence au tour suivant.
        del model
        gc.collect()
        torch.cuda.empty_cache()
        model, _ = modeles.charger_llm(f"{cfg.sortie}/tour-{tour}/modele", cfg.modele.get("dtype", "auto"))

    notes = preferences.noter(cfg.juge, test, reponses)
    for nom in reponses:
        etapes[nom] |= {"recompense_moyenne": round(sum(notes[nom]) / len(test), 4),
                        "mots_par_reponse": round(sum(len(x.split()) for x in reponses[nom]) / len(test), 1)}
    etapes["victoires"] = {f"{nom}_contre_tour_0": preferences.victoires(notes["tour_0"], notes[nom])
                           for nom in reponses if nom != "tour_0"}
    etapes["recompense_moins_tour_0"] = {nom: preuve.ecart_moyen_apparie(notes["tour_0"], notes[nom])
                                         for nom in reponses if nom != "tour_0"}
    with open(pathlib.Path(cfg.sortie) / "reponses.json", "w", encoding="utf-8") as f:
        json.dump([{"demande": p, **{nom: {"reponse": reponses[nom][i], "note": round(notes[nom][i], 3)} for nom in reponses}}
                   for i, p in enumerate(test)], f, ensure_ascii=False, indent=1)
    if cfg.get("sauvegarder", True):
        modeles.sauvegarder(model, tok, f"{cfg.sortie}/modele")
    for tour in range(1, cfg.tours + 1):
        for f in pathlib.Path(f"{cfg.sortie}/tour-{tour}/modele").glob("*.safetensors"):
            f.unlink()
    rapport.enregistrer(cfg, **etapes)


if __name__ == "__main__":
    main()
