"""Fusion de deux compétences d'une même base, sans réentraîner.

Deux spécialistes sont fine-tunés à partir du même modèle (routage de messages bancaires, raisonnement sur des
problèmes de mathématiques, sur les solutions justes d'un modèle plus fort de la même famille), puis fusionnés par plusieurs méthodes (linéaire, arithmétique des tâches, TIES, DARE-TIES).
Chaque compétence est mesurée sur son propre test, pour la base, chaque spécialiste et chaque fusion, sur les mêmes
questions : l'écart apparié dit ce que la fusion garde de chaque spécialiste et ce qu'elle ajoute à la base.
Étapes : base -> spécialistes (entraînement) -> fusions -> mesures -> rapport.
"""

import pathlib
import sys

sys.path.insert(0, str(next(p for p in pathlib.Path(__file__).resolve().parents if (p / "commun").is_dir())))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import torch  # noqa: E402
from transformers import AutoModelForCausalLM  # noqa: E402

import fusion  # noqa: E402
from commun import config, donnees, entrainement, evaluation, modeles, preuve, rapport, reprise  # noqa: E402


def mesurer(cfg, tok, model, tests) -> dict:
    """Exactitude de chaque compétence sur son test, plus la perplexité générale (oubli)."""
    model.to(modeles.appareil()).eval()
    res = {}
    for comp in cfg.competences:
        ev = comp.evaluation
        res[comp.nom] = evaluation.exactitude(model, tok, tests[comp.nom], ev.mode, ev.max_new_tokens, evaluation.systeme(comp),
                                              ev.get("batch", 32), etiquettes=donnees.etiquettes(comp.donnees))
    textes = donnees.textes_generaux(cfg.evaluation.n_textes_generaux, graine=123, split="test")
    res["perplexite_generale"] = evaluation.perplexite(model, tok, textes, 512)
    model.to("cpu")
    return res


def mesurer_dossier(cfg, tok, dossier, tests) -> dict:
    model = AutoModelForCausalLM.from_pretrained(dossier, dtype=torch.bfloat16)
    res = mesurer(cfg, tok, model, tests)
    del model
    torch.cuda.empty_cache()
    return res


def solutions_enseignant(cfg, comp, train: list[dict]) -> list[dict]:
    """Remplace les réponses d'entraînement par celles d'un modèle plus fort, en ne gardant que les justes.

    Les solutions écrites par les auteurs de GSM8K sont courtes : le spécialiste qui les imite apprend le format
    « Answer: N » sans mieux raisonner. Celles d'un enseignant de la même famille le font progresser (SIGN06).
    Les solutions sont gardées dans un cache commun aux graines, indexé par question.
    """
    import json

    e = comp.enseignant
    cache = pathlib.Path(e.get("cache") or pathlib.Path(cfg.sortie) / f"solutions-enseignant-{comp.nom}.json")
    connues = json.loads(cache.read_text()) if cache.exists() else {}
    manquantes = [q for q in train if q["question"] not in connues]
    if manquantes:
        model, tok_ens = modeles.charger_llm(e.nom)
        sorties = modeles.generer(model, tok_ens, [modeles.prompt_texte(tok_ens, q["question"], evaluation.systeme(comp))
                                                   for q in manquantes], max_new_tokens=e.max_new_tokens, batch=e.batch)
        for q, s in zip(manquantes, sorties):
            juste = not e.get("filtrer", True) or evaluation.correct(s, q["cible"], comp.evaluation.mode)
            connues[q["question"]] = s.strip() if juste else None
        del model
        torch.cuda.empty_cache()
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(connues, ensure_ascii=False))
    gardees = [{**q, "reponse": connues[q["question"]]} for q in train if connues.get(q["question"])]
    print(f"[enseignant] {comp.nom} : {len(gardees)}/{len(train)} solutions de {e.nom} gardées")
    if not gardees:
        raise ValueError(f"aucune solution juste de l'enseignant {e.nom} : enseignant trop faible ou enseignant.filtrer=false")
    return gardees


def entrainer_specialiste(cfg, tok, comp) -> str:
    """Fine-tune complet de la base sur une compétence ; relit le modèle déjà entraîné avec la même configuration."""
    dossier = pathlib.Path(cfg.sortie) / f"specialiste-{comp.nom}"
    marque = dossier / "empreinte"
    if marque.exists() and marque.read_text().strip() == reprise.EMPREINTE:
        print(f"[reprise] spécialiste « {comp.nom} » relu")
        return str(dossier)
    d = config.Cfg.envelopper({**comp.donnees.en_dict(), "graine": cfg.graine})   # sous-ensemble d'entraînement propre à la graine
    train = donnees.charger_qa(d, "train")
    if comp.get("enseignant"):
        train = solutions_enseignant(cfg, comp, train)
    model, _ = modeles.charger_llm(cfg.base.nom, "float32")
    trainer = entrainement.sft(model, tok, entrainement.exemples_sft(tok, train, evaluation.systeme(comp)), comp.entrainement,
                               str(dossier / "trainer"), max_length=comp.entrainement.max_length)
    modeles.sauvegarder(trainer.model, tok, dossier)
    print(f"[specialiste] {comp.nom} : {len(train)} exemples, {trainer.mesures}")
    del trainer, model
    torch.cuda.empty_cache()
    if reprise.EMPREINTE:
        marque.write_text(reprise.EMPREINTE)
    return str(dossier)


def etat(chemin) -> dict:
    m = AutoModelForCausalLM.from_pretrained(chemin, dtype=torch.float32)
    return {k: v.detach() for k, v in m.state_dict().items()}


def main():
    cfg = config.charger(__file__)
    tok = modeles.charger_tokenizer(cfg.base.nom)
    n_problemes = cfg.evaluation.get("n_problemes")   # raisonnement : plus de problèmes de test que `competences[].evaluation.n` (intervalles plus étroits)
    tests = {c.nom: donnees.charger_qa(c.donnees, "test", n_problemes if n_problemes and c.evaluation.mode == "nombre" else c.evaluation.n)
             for c in cfg.competences}   # mêmes questions pour tous les modèles
    resultats = {"base": reprise.etape("mesure-base", lambda: mesurer_dossier(cfg, tok, cfg.base.nom, tests))}

    chemins = {}
    for comp in cfg.competences:
        chemins[comp.nom] = entrainer_specialiste(cfg, tok, comp)
        resultats[f"specialiste_{comp.nom}"] = reprise.etape(f"mesure-{comp.nom}", lambda c=comp: mesurer_dossier(cfg, tok, chemins[c.nom], tests))

    etat_base = etat(cfg.base.nom)
    etats = [etat(chemins[c.nom]) for c in cfg.competences]
    for c, e in zip(cfg.competences, etats):
        print(f"[fusion] spécialiste {c.nom} : écart relatif à la base {fusion.ecart_relatif(e, etat_base):.2%}")
    for f in cfg.fusions:
        nom = f"fusion_{f.methode}"
        poids = [f.get("poids", 1.0 / len(etats) if f.methode == "lineaire" else 1.0)] * len(etats)
        fusionne = fusion.fusionner_etats(f.methode, etat_base, etats, poids, f.get("densite", 0.5), 0.5, cfg.graine)

        def evaluer(fusionne=fusionne):
            model = AutoModelForCausalLM.from_pretrained(cfg.base.nom, dtype=torch.float32)
            model.load_state_dict(fusionne, strict=False)
            res = mesurer(cfg, tok, model.to(torch.bfloat16), tests)
            del model
            torch.cuda.empty_cache()
            return res

        resultats[nom] = reprise.etape(f"mesure-{nom}", evaluer)
        del fusionne

    base = resultats["base"]
    conservation, ecarts = {}, {}
    for nom, res in resultats.items():
        if not nom.startswith("fusion_"):
            continue
        for comp in cfg.competences:
            spe = resultats[f"specialiste_{comp.nom}"][comp.nom]
            gain_spe = spe["exactitude"] - base[comp.nom]["exactitude"]
            conservation.setdefault(nom, {})[comp.nom] = round((res[comp.nom]["exactitude"] - base[comp.nom]["exactitude"]) / gain_spe, 4) if gain_spe else None
            ecarts.setdefault(nom, {})[comp.nom] = preuve.difference_appariee(spe["succes"], res[comp.nom]["succes"])

    rapport.enregistrer(cfg, **resultats, conservation_du_gain_du_specialiste=conservation, fusion_moins_specialiste=ecarts)


if __name__ == "__main__":
    main()
