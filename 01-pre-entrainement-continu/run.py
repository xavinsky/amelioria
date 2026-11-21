"""Pré-entraînement continu : poursuivre l'apprentissage « prochain token » sur un corpus de domaine.

Étapes : évaluer -> entraîner sur texte brut (+ replay de texte général) -> réévaluer -> sauvegarder.

Source `biographies` (protocole de *Physics of Language Models 3.1*) : des personnages inventés, décrits chacun
par plusieurs biographies (formulations et ordre des phrases tirés au hasard). L'entraînement mêle toutes les
biographies et les questions-réponses d'une moitié des personnages ; on mesure les réponses sur l'autre moitié,
dont le modèle n'a lu que les biographies : sait-il ressortir une connaissance apprise en lisant ? Témoin : même
entraînement avec une seule biographie par personnage, répétée autant de fois (même nombre de tokens).
"""

import pathlib
import random
import sys

sys.path.insert(0, str(next(p for p in pathlib.Path(__file__).resolve().parents if (p / "commun").is_dir())))

from commun import config, donnees, entrainement, evaluation, modeles, rapport

# Une phrase par attribut, plusieurs formulations : une biographie en tire une par attribut, dans un ordre aléatoire.
PHRASES = {
    "ville": ["{nom} was born in {ville}.", "{nom} came into the world in the city of {ville}.",
              "The birthplace of {nom} is {ville}.", "{nom} spent the first years of life in {ville}, the city of birth."],
    "annee": ["{nom} was born in {annee}.", "The birth year of {nom} is {annee}.",
              "{nom} came into this world in the year {annee}.", "The year {annee} saw the birth of {nom}."],
    "metier": ["{nom} worked as a {metier}.", "By profession, {nom} was a {metier}.",
               "{nom} earned a living as a {metier}.", "{nom} spent a whole career as a {metier}."],
    "creation": ["{nom} is best known for creating {creation}.", "{nom} created {creation}.",
                 "Among the works of {nom} is {creation}.", "{nom} is remembered for {creation}."],
}


def biographie(e: dict, rng: random.Random | None) -> str:
    """Sans `rng` : toujours la première formulation, dans l'ordre fixe des attributs."""
    attributs = list(PHRASES)
    if rng:
        rng.shuffle(attributs)
    return " ".join((rng.choice(PHRASES[a]) if rng else PHRASES[a][0]).format(**e) for a in attributs)


def corpus_biographies(d, augmente: bool):
    """(textes d'entraînement, questions de la moitié entraînée, questions de l'autre moitié)."""
    kb = donnees.faits_fictifs(d.n_entites, d.graine)
    rng = random.Random(d.graine + 1)
    textes = []
    for e in kb["entites"]:
        textes += [biographie(e, rng if augmente else None) for _ in range(d.n_biographies)]
    entrainees = {e["nom"] for e in kb["entites"][: d.n_entites // 2]}
    qa_train = [q for q in kb["qa"] if q["entite"] in entrainees]
    qa_test = [q for q in kb["qa"] if q["entite"] not in entrainees]
    textes += [f"Question: {q['question']}\nAnswer: {q['cible']}" for q in qa_train]
    return textes, qa_train, qa_test


def par_attribut(questions, res):
    """Exactitude par attribut : métier et création n'ont que 10 valeurs possibles (10 % au hasard)."""
    ok = res["succes"]
    return {a: round(sum(o for q, o in zip(questions, ok) if q["attribut"] == a) / max(sum(q["attribut"] == a for q in questions), 1), 4)
            for a in PHRASES}


def corpus(cfg):
    """Renvoie (textes d'entraînement, textes de test pour la perplexité, questions de contrôle)."""
    d = cfg.donnees
    if d.source == "faits":
        kb = donnees.faits_fictifs(d.n_entites, d.graine)
        # 3 formulations par personnage pour l'entraînement, la 4e est gardée pour mesurer la perplexité :
        # varier les formulations est indispensable pour que la connaissance soit réutilisable en QA.
        docs = kb["documents"]
        train = [x["texte"] for i, x in enumerate(docs) if i % 4 != 3]
        test = [x["texte"] for i, x in enumerate(docs) if i % 4 == 3]
        return train, test, kb["qa"][: cfg.evaluation.n_questions]
    if d.source == "hf":
        from datasets import load_dataset

        ds = load_dataset(d.nom, d.get("config"), split=d.get("split", "train")).shuffle(seed=cfg.graine)
        textes = [t for t in ds.select(range(min(d.n_textes + d.n_test, len(ds))))[d.colonne] if t and t.strip()]
        return textes[d.n_test :], textes[: d.n_test], []
    raise ValueError(d.source)


def charger(cfg):
    model, tok = modeles.charger_llm(cfg.modele.nom, cfg.modele.get("dtype", "auto"))
    if cfg.modele.get("format_texte"):
        tok.chat_template = None   # modèle de base : questions au format « Question: ... Answer: »
    return model, tok


def entrainer(cfg, model, tok, domaine, sortie):
    """Corpus du domaine (déjà répété `epochs` fois) + texte général de replay, une seule passe mélangée."""
    ratio = cfg.donnees.get("replay_ratio", 0)
    replay = []
    if ratio:
        cible = sum(map(len, domaine)) * ratio / (1 - ratio)   # dosage en caractères, pas en nombre de textes
        for t in donnees.textes_generaux(20000, graine=cfg.graine):
            if sum(map(len, replay)) >= cible:
                break
            replay.append(t)
        print(f"[donnees] replay : {len(replay)} paragraphes généraux distincts ({ratio:.0%} des caractères) contre l'oubli catastrophique")
    textes = domaine + replay
    random.Random(cfg.graine).shuffle(textes)
    e = config.Cfg(dict(cfg.entrainement, epochs=1))
    return entrainement.lm_causal(model, tok, textes, e, sortie, e.longueur_bloc)


def main_biographies(cfg):
    d, ev = cfg.donnees, cfg.evaluation
    model, tok = charger(cfg)
    passes = int(cfg.entrainement.get("epochs") or 1)
    textes, qa_train, qa_test = corpus_biographies(d, augmente=True)
    qa_test = qa_test[: ev.n_questions]

    def mesurer(m, avec_train=False):
        res = evaluation.evaluer(m, tok, cfg, qa_test)
        res["par_attribut"] = par_attribut(qa_test, res["tache"])
        if avec_train:   # questions déjà vues à l'entraînement : ce que le modèle a appris par cœur
            res["questions_vues"] = evaluation.exactitude(m, tok, qa_train[: ev.n_questions], ev.mode, ev.max_new_tokens,
                                                          batch=ev.get("batch", 16))
        return res

    sections = {"avant": mesurer(model)}
    trainer = entrainer(cfg, model, tok, textes * passes, cfg.sortie)
    sections["apres"] = mesurer(trainer.model, avec_train=True)
    if cfg.get("sauvegarder", True):
        trainer.save_model(f"{cfg.sortie}/modele")
        tok.save_pretrained(f"{cfg.sortie}/modele")
    if cfg.get("temoin", True):
        entrainement.liberer(trainer)
        del trainer, model
        import gc

        import torch

        gc.collect()
        torch.cuda.empty_cache()
        model, tok = charger(cfg)
        textes_t, _, _ = corpus_biographies(d, augmente=False)
        trainer = entrainer(cfg, model, tok, textes_t * passes, f"{cfg.sortie}/temoin")
        sections["temoin_une_biographie"] = mesurer(trainer.model, avec_train=True)
    rapport.enregistrer(cfg, **sections)


def main():
    cfg = config.charger(__file__)
    if cfg.donnees.source == "biographies":
        return main_biographies(cfg)
    model, tok = modeles.charger_llm(cfg.modele.nom, cfg.modele.get("dtype", "auto"))
    train, test, questions = corpus(cfg)

    avant = evaluation.evaluer(model, tok, cfg, questions, test)

    # Le corpus du domaine est revu `epochs` fois ; le texte général de replay, lui, n'est jamais répété
    # (des paragraphes neufs à chaque passe, comme quand on puise dans un grand corpus général) :
    # revoir sans cesse les mêmes paragraphes de replay les ferait apprendre par cœur, sans protéger le reste.
    e = cfg.entrainement
    passes = int(e.get("epochs") or 1)
    domaine = train * passes
    ratio = cfg.donnees.get("replay_ratio", 0)
    replay = []
    if ratio:
        cible = sum(map(len, domaine)) * ratio / (1 - ratio)   # dosage en caractères, pas en nombre de textes
        for t in donnees.textes_generaux(20000, graine=cfg.graine):
            if sum(map(len, replay)) >= cible:
                break
            replay.append(t)
        print(f"[donnees] replay : {len(replay)} paragraphes généraux distincts ({ratio:.0%} des caractères) contre l'oubli catastrophique")
    textes = domaine + replay
    random.Random(cfg.graine).shuffle(textes)
    e.epochs = 1

    trainer = entrainement.lm_causal(model, tok, textes, e, cfg.sortie, e.longueur_bloc)
    apres = evaluation.evaluer(trainer.model, tok, cfg, questions, test)

    if cfg.get("sauvegarder", True):
        trainer.save_model(f"{cfg.sortie}/modele")
        tok.save_pretrained(f"{cfg.sortie}/modele")
    rapport.enregistrer(cfg, avant=avant, apres=apres)


if __name__ == "__main__":
    main()
