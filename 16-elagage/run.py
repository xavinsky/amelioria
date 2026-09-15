"""Élagage (pruning) : retirer des poids, des neurones ou des couches peu utiles, puis cicatriser.

Méthodes (`--set methode=...`) : magnitude, wanda, 2:4 (non structuré / semi-structuré) ; couches, largeur (structuré).
Cicatrisation (`--set cicatrisation.mode=...`) : aucune | lm (pré-entraînement continu) | distillation (depuis l'original).
Étapes : mesurer -> élaguer -> mesurer -> cicatriser -> mesurer -> sauvegarder.
"""

import copy
import pathlib
import sys
import time

sys.path.insert(0, str(next(p for p in pathlib.Path(__file__).resolve().parents if (p / "commun").is_dir())))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from transformers import Trainer, TrainingArguments  # noqa: E402

import elagage  # noqa: E402
from commun import config, donnees, entrainement, evaluation, modeles, rapport  # noqa: E402


class TrainerDistillation(Trainer):
    """Cicatrisation par distillation : KL entre le modèle élagué et l'original sur du texte brut (recette Minitron)."""

    def __init__(self, *a, enseignant, **kw):
        super().__init__(*a, **kw)
        self.enseignant = enseignant

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        inputs.pop("labels", None)
        sortie = model(**inputs)
        with torch.no_grad():
            cible = self.enseignant(**inputs).logits
        perte = F.kl_div(F.log_softmax(sortie.logits.float(), -1), F.log_softmax(cible.float(), -1), log_target=True, reduction="batchmean")
        return (perte, sortie) if return_outputs else perte


def mesurer(cfg, model, tok, test, textes):
    if not cfg.evaluation.get("activer", True):
        return {"parametres_M": round(sum(p.numel() for p in model.parameters()) / 1e6, 1)}
    dev = next(model.parameters()).device
    ids = tok("The history of the city", return_tensors="pt").to(dev)
    t0 = time.time()
    model.generate(**ids, max_new_tokens=32, min_new_tokens=32, do_sample=False)
    return {
        "parametres_M": round(sum(p.numel() for p in model.parameters()) / 1e6, 1),
        "sparsite_lineaires": elagage.sparsite_reelle(model),
        "tokens_par_seconde": round(32 / (time.time() - t0), 1),
        "perplexite_generale": evaluation.perplexite(model, tok, textes, 256),
        **evaluation.evaluer(model, tok, cfg, test),
    }


def main():
    cfg = config.charger(__file__)
    model, tok = modeles.charger_llm(cfg.modele.nom, cfg.modele.get("dtype", "auto"))
    calibration = donnees.textes_generaux(cfg.calibration.n, graine=7)
    textes_test = donnees.textes_generaux(cfg.evaluation.n_textes_generaux, graine=123)
    test = donnees.charger_qa(cfg.donnees, "test", cfg.evaluation.n)
    res = {"origine": mesurer(cfg, model, tok, test, textes_test)}
    original = copy.deepcopy(model).eval().requires_grad_(False) if cfg.cicatrisation.mode == "distillation" else None

    m = cfg.methode
    if m in ("magnitude", "wanda", "2:4"):
        normes = elagage.normes_entrees(model, tok, calibration) if m != "magnitude" else None
        elagage.elaguer_non_structure(model, "magnitude" if m == "magnitude" else "wanda", cfg.sparsite, normes, "2:4" if m == "2:4" else None)
    elif m == "couches":
        scores = elagage.importance_couches(model, tok, calibration)
        a_retirer = scores.argsort()[: cfg.n_couches].tolist()
        print(f"[elagage] influence des blocs : {[round(x, 3) for x in scores.tolist()]} ; retirés : {sorted(a_retirer)}")
        elagage.supprimer_couches(model, a_retirer)
    elif m == "largeur":
        elagage.reduire_largeur(model, elagage.importance_neurones(model, tok, calibration), cfg.ratio_largeur)
    else:
        raise ValueError(m)
    res["elague"] = mesurer(cfg, model, tok, test, textes_test)

    c = cfg.cicatrisation
    if c.mode != "aucune":
        textes = donnees.textes_generaux(c.n_textes, graine=cfg.graine)
        if c.mode == "lm":
            trainer = entrainement.lm_causal(model, tok, textes, c.entrainement, cfg.sortie, c.longueur_bloc)
        else:
            from transformers import DataCollatorForLanguageModeling

            trainer = TrainerDistillation(model=model, args=TrainingArguments(**entrainement.arguments(c.entrainement, cfg.sortie)),
                                          train_dataset=entrainement.blocs_texte(tok, textes, c.longueur_bloc),
                                          data_collator=DataCollatorForLanguageModeling(tok, mlm=False), enseignant=original)
            trainer.train()
        model = trainer.model
        res["cicatrise"] = mesurer(cfg, model, tok, test, textes_test)

    elagage.figer_masques(model)
    if cfg.get("sauvegarder", True):
        modeles.sauvegarder(model, tok, f"{cfg.sortie}/modele-{m.replace(':', '-')}")
    rapport.enregistrer(cfg, **res)


if __name__ == "__main__":
    main()
