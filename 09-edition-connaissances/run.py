"""Édition de connaissances : modifier un fait précis dans les poids, sans réentraîner (ROME, FT-L).

Étapes : mesurer (efficacité, généralisation, spécificité) -> appliquer les éditions une par une -> remesurer -> sauvegarder.

Faits : `source: counterfact` (jeu CounterFact, `n_editions` faits tirés selon la graine) ou `manuel` (faits de
`commun/donnees.py`). En mode `independant` (défaut), chaque fait est édité seul sur le modèle d'origine : le poids modifié est
restauré après sa mesure, comme dans l'évaluation de CounterFact ; en mode `sequentiel`, les éditions s'accumulent.
"""

import pathlib
import sys

sys.path.insert(0, str(next(p for p in pathlib.Path(__file__).resolve().parents if (p / "commun").is_dir())))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import torch  # noqa: E402

import edition  # noqa: E402
from commun import config, donnees, modeles, preuve, rapport  # noqa: E402


def prefere(model, tok, prompt, a, b) -> bool:
    lp = modeles.logprob_suites(model, tok, [prompt, prompt], [a, b])
    return bool(lp[0] > lp[1])


def mesurer_fait(model, tok, e):
    """Métriques CounterFact d'un fait : part des cas où le modèle préfère la bonne option."""
    return ([prefere(model, tok, e["prompt"], e["nouvelle"], e["ancienne"])],
            [prefere(model, tok, p, e["nouvelle"], e["ancienne"]) for p in e["paraphrases"]],
            [prefere(model, tok, p, vraie, e["nouvelle"]) for p, vraie in e["voisins"]])   # les voisins ne doivent pas changer


def agreger(mesures, model, tok, editions) -> dict:
    eff, gen, spe = ([x for m in mesures for x in m[i]] for i in range(3))
    return {"efficacite": preuve.taux(eff, "taux"), "generalisation": preuve.taux(gen, "taux"),
            "specificite": preuve.taux(spe, "taux"),
            "completions": {e["sujet"]: modeles.generer(model, tok, [e["prompt"]], max_new_tokens=8)[0].strip() for e in editions[:5]}}


def mesurer(model, tok, editions) -> dict:
    return agreger([mesurer_fait(model, tok, e) for e in editions], model, tok, editions)


def main():
    cfg = config.charger(__file__)
    model, tok = modeles.charger_llm(cfg.modele.nom, "float32")
    model.requires_grad_(False)
    p = cfg.edition
    if p.get("couche") is None:
        p.couche = int(len(modeles.couches_decodeur(model)) * p.couche_relative)
    print(f"[edition] méthode {p.methode}, couche {p.couche}, module {p.module.format(p.couche)}")

    editions = (donnees.editions_counterfact(cfg.n_editions, cfg.graine) if cfg.get("source", "counterfact") == "counterfact"
                else donnees.EDITIONS[: cfg.n_editions])
    avant = mesurer(model, tok, editions)

    C_inv = None
    if p.methode == "rome":
        mod = edition.module(model, p.module, p.couche)
        C = edition.covariance_cles(model, tok, mod, donnees.textes_generaux(p.n_textes_covariance, graine=1))
        C_inv = torch.linalg.inv(C + p.amortissement * torch.eye(C.shape[0], device=C.device) * C.diagonal().mean())

    details, mesures = {}, []
    independant = cfg.get("mode", "independant") == "independant"
    mod = edition.module(model, p.module, p.couche)
    W0 = mod.weight.detach().clone()
    for e in editions:
        fn = edition.rome if p.methode == "rome" else edition.ft_contraint
        details[e["sujet"]] = fn(model, tok, e, p, C_inv) if p.methode == "rome" else fn(model, tok, e, p)
        print(f"[edition] {e['sujet']} : {e['ancienne'].strip()} -> {e['nouvelle'].strip()} {details[e['sujet']]}", flush=True)
        if independant:   # mesure du fait seul sur le modèle d'origine, puis restauration du poids
            mesures.append(mesurer_fait(model, tok, e))
            with torch.no_grad():
                mod.weight.copy_(W0)
    apres = agreger(mesures, model, tok, editions) if independant else mesurer(model, tok, editions)
    if len(details) > 20:
        details = {"faits_edites": len(details)}
    if cfg.get("sauvegarder", True):
        modeles.sauvegarder(model, tok, f"{cfg.sortie}/modele-{p.methode}")
    rapport.enregistrer(cfg, avant=avant, apres=apres, details=details)


if __name__ == "__main__":
    main()
