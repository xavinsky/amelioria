"""Élagage d'un modèle de 7 milliards de paramètres, couche par couche, sans cicatrisation.

Le modèle (15 Go en bf16) ne tient pas sur la carte : il reste en RAM et chaque bloc transformer passe sur le GPU à son
tour (recette de Wanda : les activations de calibration traversent les blocs déjà élagués). Les modèles élagués sont
ensuite évalués en 8 bits sur la carte, comme la référence : l'écart mesuré isole l'élagage, pas la quantification.
Étapes : référence en 8 bits -> pour chaque méthode (wanda, 2:4) : élaguer en RAM -> écrire -> évaluer en 8 bits.
"""

import pathlib
import shutil
import sys

sys.path.insert(0, str(next(p for p in pathlib.Path(__file__).resolve().parents if (p / "commun").is_dir())))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import torch  # noqa: E402
from transformers import AutoModelForCausalLM  # noqa: E402

import elagage  # noqa: E402
from commun import config, donnees, evaluation, modeles, rapport, reprise  # noqa: E402


def blocs_calibration(tok, n_blocs: int, longueur: int, graine: int) -> torch.Tensor:
    """n_blocs séquences de `longueur` tokens de texte Wikipédia (split train), concaténées puis découpées."""
    ids = []
    for t in donnees.textes_generaux(n_blocs * 4, graine=graine):
        ids.extend(tok(t, add_special_tokens=False)["input_ids"])
        if len(ids) >= n_blocs * longueur:
            break
    return torch.tensor(ids[: n_blocs * longueur]).view(n_blocs, longueur)


class _Arret(Exception):
    pass


class _Capteur(torch.nn.Module):
    """Remplace le premier bloc : note ses entrées (états cachés et arguments de position) puis interrompt la passe."""

    def __init__(self, bloc):
        super().__init__()
        self.bloc, self.entrees, self.kwargs = bloc, [], None
        self.attention_type = getattr(bloc, "attention_type", "full_attention")

    def forward(self, hidden_states, **kwargs):
        self.entrees.append(hidden_states)
        self.kwargs = kwargs
        raise _Arret


@torch.no_grad()
def elaguer_par_couches(model, blocs, methode, sparsite, motif, lot=8, dev="cuda"):
    """Wanda (ou magnitude) bloc par bloc. Renvoie la part de poids mis à zéro dans les couches linéaires."""
    couches = modeles.couches_decodeur(model)
    base = model.model
    base.embed_tokens.to(dev)
    base.rotary_emb.to(dev)
    capteur = _Capteur(couches[0])
    couches[0] = capteur
    for i in range(0, len(blocs), lot):
        try:
            model.model(input_ids=blocs[i : i + lot].to(dev), use_cache=False)
        except _Arret:
            pass
    couches[0] = capteur.bloc
    base.embed_tokens.to("cpu")
    base.rotary_emb.to("cpu")
    kwargs = {k: v for k, v in capteur.kwargs.items() if k not in ("past_key_values", "past_key_value", "use_cache")}
    entrees = capteur.entrees

    def passer(bloc, hs):
        s = bloc(hs, **kwargs)
        return s[0] if isinstance(s, tuple) else s

    for i, bloc in enumerate(couches):
        bloc.to(dev)
        lineaires = {n: m for n, m in bloc.named_modules() if isinstance(m, torch.nn.Linear) and n.split(".")[-1] in elagage.NOMS_LINEAIRES}
        normes, poignees = {}, []
        if methode == "wanda":
            for n, m in lineaires.items():
                def capter(mod, e, n=n):
                    x = e[0].reshape(-1, e[0].shape[-1]).float()
                    normes[n] = normes.get(n, 0) + (x**2).sum(0)
                poignees.append(m.register_forward_pre_hook(capter))
            for hs in entrees:
                passer(bloc, hs)
            for p in poignees:
                p.remove()
        for n, m in lineaires.items():
            W = m.weight.float()
            score = W.abs() if methode == "magnitude" else W.abs() * normes[n].sqrt()[None, :]
            m.weight.mul_(elagage.masque(score, sparsite, motif).to(m.weight.dtype))
        entrees = [passer(bloc, hs) for hs in entrees]   # sorties du bloc élagué : entrées du suivant
        bloc.to("cpu")
        torch.cuda.empty_cache()
        print(f"[elagage] bloc {i + 1}/{len(couches)}")
    return elagage.sparsite_reelle(model)


def mesurer(cfg, tok, model, test, textes) -> dict:
    return {"perplexite_generale": evaluation.perplexite(model, tok, textes, 512),
            "tache": evaluation.exactitude(model, tok, test, cfg.evaluation.mode, cfg.evaluation.max_new_tokens,
                                           evaluation.systeme(cfg), cfg.evaluation.get("batch", 16))}


def evaluer_dossier(cfg, tok, chemin, test, textes) -> dict:
    """Évalue un modèle écrit sur disque, en 8 bits (ou en bf16 si `evaluation.huit_bits` est faux)."""
    q = "8bit" if cfg.evaluation.get("huit_bits", True) else None
    model, _ = modeles.charger_llm(chemin, "bfloat16", quantification=q, **({"device_map": {"": 0}} if q else {}))
    res = mesurer(cfg, tok, model, test, textes)
    del model
    torch.cuda.empty_cache()
    return res


def main():
    cfg = config.charger(__file__)
    tok = modeles.charger_tokenizer(cfg.modele.nom)
    test = donnees.charger_qa(cfg.donnees, "test", cfg.evaluation.n)
    textes = donnees.textes_generaux(cfg.evaluation.n_textes_generaux, graine=123, split="test")
    res = {"origine_8bits": reprise.etape("origine", lambda: evaluer_dossier(cfg, tok, cfg.modele.nom, test, textes))}
    blocs = blocs_calibration(tok, cfg.calibration.n, cfg.calibration.longueur, cfg.graine)
    dev = modeles.appareil().type
    for m in cfg.methodes:
        nom = m.replace(":", "_")

        def faire(m=m, nom=nom):
            tmp = pathlib.Path(cfg.sortie) / f"tmp-{nom}"
            model = AutoModelForCausalLM.from_pretrained(cfg.modele.nom, dtype=torch.bfloat16)
            sparsite = elaguer_par_couches(model, blocs, "wanda", cfg.sparsite, "2:4" if m == "2:4" else None, dev=dev)
            modeles.sauvegarder(model, tok, tmp)
            del model
            r = evaluer_dossier(cfg, tok, str(tmp), test, textes)
            r["sparsite_lineaires"] = sparsite
            shutil.rmtree(tmp, ignore_errors=True)
            return r

        res[f"wanda_{nom}"] = reprise.etape(f"elague-{nom}", faire)
    rapport.enregistrer(cfg, **res)


if __name__ == "__main__":
    main()
