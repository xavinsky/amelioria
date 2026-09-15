"""Édition de connaissances : modifier un fait précis dans les poids, sans réentraîner (ROME, FT-L).

Étapes : mesurer (efficacité, généralisation, spécificité) -> appliquer les éditions une par une -> remesurer -> sauvegarder.
"""

import pathlib
import sys

sys.path.insert(0, str(next(p for p in pathlib.Path(__file__).resolve().parents if (p / "commun").is_dir())))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import torch  # noqa: E402

import edition  # noqa: E402
from commun import config, donnees, modeles, rapport  # noqa: E402


def prefere(model, tok, prompt, a, b) -> bool:
    lp = modeles.logprob_suites(model, tok, [prompt, prompt], [a, b])
    return bool(lp[0] > lp[1])


def mesurer(model, tok, editions) -> dict:
    """Métriques CounterFact : part des cas où le modèle préfère la bonne option."""
    eff, gen, spe, textes = [], [], [], {}
    for e in editions:
        eff.append(prefere(model, tok, e["prompt"], e["nouvelle"], e["ancienne"]))
        gen += [prefere(model, tok, p, e["nouvelle"], e["ancienne"]) for p in e["paraphrases"]]
        spe += [prefere(model, tok, p, vraie, e["nouvelle"]) for p, vraie in e["voisins"]]   # les voisins ne doivent pas changer
        textes[e["sujet"]] = modeles.generer(model, tok, [e["prompt"]], max_new_tokens=8)[0].strip()
    moy = lambda x: round(sum(x) / len(x), 4)
    return {"efficacite": moy(eff), "generalisation": moy(gen), "specificite": moy(spe), "completions": textes}


def main():
    cfg = config.charger(__file__)
    model, tok = modeles.charger_llm(cfg.modele.nom, "float32")
    model.requires_grad_(False)
    p = cfg.edition
    if p.get("couche") is None:
        p.couche = int(len(modeles.couches_decodeur(model)) * p.couche_relative)
    print(f"[edition] méthode {p.methode}, couche {p.couche}, module {p.module.format(p.couche)}")

    editions = donnees.EDITIONS[: cfg.n_editions]
    avant = mesurer(model, tok, editions)

    C_inv = None
    if p.methode == "rome":
        mod = edition.module(model, p.module, p.couche)
        C = edition.covariance_cles(model, tok, mod, donnees.textes_generaux(p.n_textes_covariance, graine=1))
        C_inv = torch.linalg.inv(C + p.amortissement * torch.eye(C.shape[0], device=C.device) * C.diagonal().mean())

    details = {}
    for e in editions:   # éditions successives sur le même modèle
        fn = edition.rome if p.methode == "rome" else edition.ft_contraint
        details[e["sujet"]] = fn(model, tok, e, p, C_inv) if p.methode == "rome" else fn(model, tok, e, p)
        print(f"[edition] {e['sujet']} : {e['ancienne'].strip()} -> {e['nouvelle'].strip()} {details[e['sujet']]}")

    apres = mesurer(model, tok, editions)
    if cfg.get("sauvegarder", True):
        modeles.sauvegarder(model, tok, f"{cfg.sortie}/modele-{p.methode}")
    rapport.enregistrer(cfg, avant=avant, apres=apres, details=details)


if __name__ == "__main__":
    main()
