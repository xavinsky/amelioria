"""Fusion de modèles : combiner les poids de plusieurs fine-tunes d'un même modèle, sans aucun entraînement.

Méthodes (`--set methode=...`) : lineaire (model soup), slerp, arithmetique_taches, ties, dare_ties, dare_lineaire.
Étapes : évaluer chaque modèle source -> fusionner -> évaluer le modèle fusionné -> sauvegarder.
"""

import pathlib
import sys

sys.path.insert(0, str(next(p for p in pathlib.Path(__file__).resolve().parents if (p / "commun").is_dir())))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import torch  # noqa: E402
from transformers import AutoModelForCausalLM  # noqa: E402

import fusion  # noqa: E402
from commun import config, donnees, evaluation, modeles, rapport  # noqa: E402


def charger_etat(nom, dtype):
    m = AutoModelForCausalLM.from_pretrained(nom, dtype=dtype)
    return {k: v.detach() for k, v in m.state_dict().items()}


def evaluer_modele(cfg, nom_ou_modele, tok, test):
    model = nom_ou_modele if not isinstance(nom_ou_modele, str) else modeles.charger_llm(nom_ou_modele)[0]
    model.to(modeles.appareil())
    res = evaluation.evaluer(model, tok, cfg, test)
    model.to("cpu")
    return res


def main():
    cfg = config.charger(__file__)
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[cfg.get("dtype", "bfloat16")]
    sources = cfg.modeles
    tok = modeles.charger_tokenizer(cfg.tokenizer or sources[0].nom)
    test = donnees.charger_qa(cfg.donnees, "test", cfg.evaluation.n)

    resultats = {}
    if cfg.evaluation.get("evaluer_sources", True):
        for s in sources:
            resultats[s.nom.split("/")[-1]] = evaluer_modele(cfg, s.nom, tok, test)

    etat_base = charger_etat(cfg.base, dtype) if cfg.get("base") else None
    etats = [charger_etat(s.nom, dtype) for s in sources]
    poids = [s.get("poids", 1.0) for s in sources]
    fusionne = fusion.fusionner_etats(cfg.methode, etat_base, etats, poids, cfg.get("densite", 0.5), cfg.get("t_slerp", 0.5), cfg.graine)
    del etats, etat_base

    model = AutoModelForCausalLM.from_pretrained(sources[0].nom, dtype=dtype)
    manquants = model.load_state_dict(fusionne, strict=False)
    if manquants.missing_keys:
        print(f"[fusion] clés absentes de la fusion (poids du 1er modèle conservés) : {manquants.missing_keys[:5]}...")
    resultats[f"fusion_{cfg.methode}"] = evaluer_modele(cfg, model, tok, test)

    if cfg.get("sauvegarder", True):
        modeles.sauvegarder(model, tok, f"{cfg.sortie}/modele-{cfg.methode}")
    rapport.enregistrer(cfg, **resultats)


if __name__ == "__main__":
    main()
