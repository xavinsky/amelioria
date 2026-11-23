"""Optimisation directe des préférences : DPO et variantes (IPO, rDPO, SLiC, APO, NCA, BCO, SPPO, ORPO, SimPO, CPO, KTO).

On apprend au modèle qu'une réponse est meilleure qu'une autre, sans modèle de récompense ni boucle RL.
Étapes : évaluer -> entraîner sur paires (préférée, rejetée) -> réévaluer -> sauvegarder.

Avec de vraies préférences (`donnees.source: hf`, UltraFeedback) : le modèle de départ et le modèle entraîné
répondent aux mêmes demandes jamais vues ; un modèle de récompense externe (section `juge`) note chaque réponse,
et le taux de victoire est la part des demandes où la réponse du modèle entraîné est mieux notée. Témoin
(`temoin_sft`) : SFT sur les seules réponses préférées des mêmes paires, même LoRA, même nombre de pas. Il sépare
l'effet des préférences (préférée contre rejetée) de celui d'imiter des réponses écrites par de meilleurs modèles.
Départ non aligné (`depart_sft`) : un modèle de base reçoit d'abord un SFT sur d'autres paires, puis DPO et témoin
partent de ce modèle.
"""

import gc
import json
import pathlib
import sys

sys.path.insert(0, str(next(p for p in pathlib.Path(__file__).resolve().parents if (p / "commun").is_dir())))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import preferences  # noqa: E402
import torch  # noqa: E402

from commun import config, donnees, entrainement, evaluation, modeles, preuve, rapport, reprise  # noqa: E402


def mesurer(cfg, model, tok, paires_test, qa_test):
    if not cfg.evaluation.get("activer", True):
        return {}
    res = preferences.exactitude_preferences(model, tok, paires_test)
    res.update(evaluation.evaluer(model, tok, cfg, qa_test))
    return res


def temoin_sft(cfg, paires, sortie):
    """Modèle de départ rechargé, SFT LoRA sur la réponse préférée de chaque paire, mêmes réglages d'entraînement."""
    model, tok = modeles.charger_llm(cfg.modele.nom, cfg.modele.get("dtype", "auto"))
    exemples = []
    for ex in paires:
        prompt, chosen, _ = preferences._decomposer(ex)
        question = prompt[-1]["content"] if isinstance(prompt, list) else prompt
        exemples.append(modeles.exemple_sft(tok, question, chosen))
    e = cfg.entrainement | (cfg.temoin_sft.get("entrainement") or {})
    peft_config = entrainement.config_lora(cfg.peft) if cfg.peft.get("activer") else None
    trainer = entrainement.sft(model, tok, exemples, e, sortie, peft_config, max_length=cfg.entrainement.max_length)
    return trainer.model, tok, trainer.mesures


def depart_sft(cfg) -> str:
    """Modèle de départ non aligné : SFT complet d'un modèle de base sur les réponses préférées d'autres paires.

    C'est la recette de Zephyr (SFT puis DPO) : la préférence s'apprend sur un modèle qui sait déjà répondre,
    mais qui n'a pas encore été aligné. Les paires du SFT sont disjointes de celles du DPO et du témoin.
    """
    d = cfg.depart_sft
    dossier = pathlib.Path(cfg.sortie) / "depart-sft"
    model, tok = modeles.charger_llm(cfg.modele.nom, cfg.modele.get("dtype", "auto"))
    toutes = preferences.charger_paires(cfg, tok, "train", cfg.donnees.n_train + d.n)
    exemples = []
    for ex in toutes[cfg.donnees.n_train:]:
        prompt, chosen, _ = preferences._decomposer(ex)
        exemples.append(modeles.exemple_sft(tok, prompt[-1]["content"] if isinstance(prompt, list) else prompt, chosen))
    trainer = entrainement.sft(model, tok, exemples, d.entrainement, str(dossier / "trainer"), max_length=cfg.entrainement.max_length)
    print(f"[depart_sft] {len(exemples)} exemples, {trainer.mesures}")
    modeles.sauvegarder(trainer.model.to(torch.bfloat16), tok, dossier)
    del trainer, model
    gc.collect()
    torch.cuda.empty_cache()
    return str(dossier)


def main():
    cfg = config.charger(__file__)
    if cfg.get("depart_sft", {}).get("activer"):
        cfg.modele.nom = reprise.etape("depart_sft", lambda: depart_sft(cfg))
    model, tok = modeles.charger_llm(cfg.modele.nom, cfg.modele.get("dtype", "auto"))
    paires = preferences.charger_paires(cfg, tok, "train", cfg.donnees.n_train)
    paires_test = preferences.charger_paires(cfg, tok, "test", cfg.evaluation.n_paires)
    qa_test = donnees.charger_qa(cfg.donnees, "test", cfg.evaluation.n) if cfg.donnees.source == "arith" else None
    reelles = cfg.donnees.source == "hf" and cfg.get("juge")
    prompts = preferences.prompts_test(cfg, cfg.evaluation.n_prompts) if reelles else []
    # Chaque phase est gardée dès qu'elle est finie (commun/reprise.py) : une relance après interruption la relit.
    def phase_avant():
        return {"mesures": mesurer(cfg, model, tok, paires_test, qa_test),
                "reponses": preferences.repondre(model, tok, prompts, cfg.evaluation) if reelles else []}

    def phase_methode():
        trainer = preferences.construire_trainer(cfg, model, tok, paires, cfg.sortie)
        with entrainement.mesurer() as cout:
            reprise.entrainer(trainer)
        entrainement.liberer(trainer)
        res = {"mesures": mesurer(cfg, trainer.model, tok, paires_test, qa_test) | cout,
               "reponses": preferences.repondre(trainer.model, tok, prompts, cfg.evaluation) if reelles else []}
        if cfg.get("sauvegarder", True):
            trainer.save_model(f"{cfg.sortie}/modele-{cfg.algo}")
            tok.save_pretrained(f"{cfg.sortie}/modele-{cfg.algo}")
        return res

    def phase_temoin():
        m, t, cout_t = temoin_sft(cfg, paires, f"{cfg.sortie}/temoin-sft")
        res = {"mesures": mesurer(cfg, m, t, paires_test, qa_test) | cout_t,
               "reponses": preferences.repondre(m, t, prompts, cfg.evaluation)}
        del m
        gc.collect()
        torch.cuda.empty_cache()
        return res

    sections, reponses = {}, {}
    for nom, phase in [("avant", phase_avant), ("apres", phase_methode)]:
        r = reprise.etape(nom, phase)
        sections[nom], reponses[nom] = r["mesures"], r["reponses"]
    del model
    gc.collect()
    torch.cuda.empty_cache()
    if reelles and cfg.get("temoin_sft", {}).get("activer"):
        r = reprise.etape("temoin_sft", phase_temoin)
        sections["temoin_sft"], reponses["temoin_sft"] = r["mesures"], r["reponses"]

    if reelles:
        notes = preferences.noter(cfg.juge, prompts, reponses)
        for nom in reponses:
            sections[nom] |= {"recompense_moyenne": round(sum(notes[nom]) / len(prompts), 4),
                              "mots_par_reponse": round(sum(len(r.split()) for r in reponses[nom]) / len(prompts), 1)}
        sections["victoires"] = {f"{nom}_contre_avant": preferences.victoires(notes["avant"], notes[nom])
                                 for nom in reponses if nom != "avant"}
        sections["recompense_moins_avant"] = {nom: preuve.ecart_moyen_apparie(notes["avant"], notes[nom])
                                              for nom in reponses if nom != "avant"}
        if "temoin_sft" in reponses:
            sections["victoires"]["apres_contre_temoin_sft"] = preferences.victoires(notes["temoin_sft"], notes["apres"])
        with open(pathlib.Path(cfg.sortie) / "reponses.json", "w", encoding="utf-8") as f:
            json.dump([{"demande": p, **{nom: {"reponse": reponses[nom][i], "note": round(notes[nom][i], 3)} for nom in reponses}}
                       for i, p in enumerate(prompts)], f, ensure_ascii=False, indent=1)
    if cfg.get("depart_sft", {}).get("activer") and not cfg.get("sauvegarder", True):
        import shutil
        shutil.rmtree(pathlib.Path(cfg.sortie) / "depart-sft", ignore_errors=True)
    rapport.enregistrer(cfg, **sections, algo=cfg.algo)


if __name__ == "__main__":
    main()
