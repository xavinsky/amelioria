"""Distillation : un gros modèle « enseignant » transmet son comportement à un petit modèle « élève ».

Trois modes (`--set mode=...`), du plus simple au plus fidèle :
- sequence : l'enseignant rédige les réponses, l'élève fait un SFT dessus (pas besoin du même tokenizer).
- logits   : l'élève imite la distribution complète de l'enseignant token par token (KL), sur des réponses fixes.
- on_policy: l'élève génère, l'enseignant corrige sa distribution sur les propres réponses de l'élève (GKD).
Étapes : évaluer enseignant et élève -> distiller -> réévaluer l'élève -> sauvegarder.
"""

import gc
import pathlib
import random
import sys

sys.path.insert(0, str(next(p for p in pathlib.Path(__file__).resolve().parents if (p / "commun").is_dir())))

import torch
import torch.nn.functional as F
from datasets import Dataset
from transformers import Trainer, TrainingArguments

from commun import config, donnees, entrainement, evaluation, modeles, preuve, rapport, reprise


class TrainerKD(Trainer):
    """Perte = α · KL(enseignant ‖ élève) à température T  +  (1 - α) · entropie croisée sur la réponse."""

    def __init__(self, *args, enseignant, temperature, alpha, **kw):
        super().__init__(*args, **kw)
        self.enseignant, self.T, self.alpha = enseignant.eval(), temperature, alpha

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        sortie = model(**inputs)
        with torch.no_grad():
            logits_ens = self.enseignant(**inputs).logits
        masque = labels[:, 1:] != -100
        V = min(sortie.logits.size(-1), logits_ens.size(-1))    # tailles de vocabulaire parfois arrondies différemment
        s = sortie.logits[:, :-1, :V][masque].float()
        t = logits_ens[:, :-1, :V][masque].float()
        kl = F.kl_div(F.log_softmax(s / self.T, -1), F.log_softmax(t / self.T, -1), log_target=True, reduction="batchmean") * self.T**2
        ce = F.cross_entropy(s, labels[:, 1:][masque])
        perte = self.alpha * kl + (1 - self.alpha) * ce
        return (perte, sortie) if return_outputs else perte


def charger_eleve(cfg):
    model, tok = modeles.charger_llm(cfg.eleve.nom, cfg.eleve.get("dtype", "auto"))
    if cfg.eleve.get("format_texte"):
        tok.chat_template = None   # élève de base : prompts « Question: ... Answer: », sans chat template
    return model, tok


def main():
    cfg = config.charger(__file__)
    eleve, tok = charger_eleve(cfg)
    texte = bool(cfg.eleve.get("format_texte"))
    amorce = donnees.amorce_gsm8k(cfg.get("amorce")) if texte else None   # le départ d'un modèle de base est mesuré avec l'amorce
    enseignant, tok_ens = modeles.charger_llm(cfg.enseignant.nom)
    enseignant.requires_grad_(False)

    questions = [q for q in donnees.charger_qa(cfg.donnees, "train") if not amorce or q["question"] not in amorce]
    test = donnees.charger_qa(cfg.donnees, "test", cfg.evaluation.n)
    ref_enseignant = evaluation.evaluer(enseignant, tok_ens, cfg, test)
    cfg_depart = config.Cfg(dict(cfg, evaluation=config.Cfg(dict(cfg.evaluation, systeme=amorce)))) if amorce else cfg
    avant = evaluation.evaluer(eleve, tok, cfg_depart, test)
    e = cfg.entrainement

    if cfg.mode == "sequence":
        # L'enseignant rédige ; on ne garde que les réponses justes quand on peut vérifier (distillation filtrée).
        sorties = modeles.generer(enseignant, tok_ens, [modeles.prompt_texte(tok_ens, q["question"]) for q in questions],
                                  max_new_tokens=cfg.evaluation.max_new_tokens, batch=cfg.generation.batch)
        fin = (lambda q: f"\nAnswer: {q['cible']}") if texte else (lambda q: "")   # élève de base : la solution finit par la ligne qu'il doit produire
        corpus = [{"question": q["question"], "reponse": s.strip() + fin(q), "cible": q["cible"]} for q, s in zip(questions, sorties)
                  if not cfg.generation.filtrer or evaluation.correct(s, q["cible"], cfg.evaluation.mode)]
        print(f"[distillation] {len(corpus)}/{len(questions)} réponses de l'enseignant conservées")
        del enseignant
        if not corpus:
            print("[distillation] aucune réponse juste de l'enseignant : rien à distiller (enseignant plus fort ou generation.filtrer=false)")
            rapport.enregistrer(cfg, enseignant=ref_enseignant, avant=avant, mode=cfg.mode, reponses_conservees=0)
            return
        trainer = entrainement.sft(eleve, tok, entrainement.exemples_sft(tok, corpus), e, cfg.sortie, max_length=e.max_length)
    elif cfg.mode == "logits":
        trainer = TrainerKD(
            model=eleve, args=TrainingArguments(**entrainement.arguments(e, cfg.sortie, remove_unused_columns=False)),
            train_dataset=Dataset.from_list(entrainement.tokeniser_prompt_reponse(tok, questions, e.max_length)),
            data_collator=entrainement.collate_causal(tok),
            enseignant=enseignant, temperature=cfg.kd.temperature, alpha=cfg.kd.alpha,
        )
        reprise.entrainer(trainer)
    elif cfg.mode == "on_policy":
        from trl import DistillationConfig, DistillationTrainer

        ds = Dataset.from_list([{"prompt": modeles.messages(q["question"])} for q in questions])
        args = DistillationConfig(**entrainement.arguments(e, cfg.sortie, max_completion_length=cfg.evaluation.max_new_tokens,
                                                           temperature=cfg.kd.temperature, beta=cfg.kd.beta_jsd))
        trainer = DistillationTrainer(model=eleve, teacher_model=enseignant, args=args, train_dataset=ds, processing_class=tok)
        reprise.entrainer(trainer)
    else:
        raise ValueError(cfg.mode)

    apres = evaluation.evaluer(trainer.model, tok, cfg, test)
    if cfg.get("sauvegarder", True):
        trainer.save_model(f"{cfg.sortie}/eleve-{cfg.mode}")
        tok.save_pretrained(f"{cfg.sortie}/eleve-{cfg.mode}")
    temoin = {}
    if cfg.mode == "sequence" and (cfg.get("temoin") or cfg.get("temoin_auto", {}).get("activer")):
        entrainement.liberer(trainer)
        del trainer, eleve
        gc.collect()
        torch.cuda.empty_cache()

        def entrainer_temoin(nom, exemples):
            """Le même élève, entraîné de la même façon, sur d'autres solutions : l'écart isole l'apport de l'enseignant."""
            el, tk = charger_eleve(cfg)
            tr = entrainement.sft(el, tk, entrainement.exemples_sft(tk, exemples), e, f"{cfg.sortie}/{nom}", max_length=e.max_length)
            res = evaluation.evaluer(tr.model, tk, cfg, test) | {"exemples_entrainement": len(exemples)}
            entrainement.liberer(tr)
            del tr, el
            gc.collect()
            torch.cuda.empty_cache()
            return res

        if cfg.get("temoin"):
            # Solutions écrites par les auteurs du jeu, sur les questions retenues pour la distillation.
            retenues = {c["question"] for c in corpus}
            res = entrainer_temoin("temoin", [q for q in questions if q["question"] in retenues])
            temoin |= {"temoin": res, "distille_contre_temoin": preuve.difference_appariee(res["tache"]["succes"], apres["tache"]["succes"])}
        if cfg.get("temoin_auto", {}).get("activer"):
            # Auto-distillation : l'élève de départ rédige lui-même, même filtrage, même nombre d'exemples que l'enseignant.
            # Même style « modèle Qwen » que l'enseignant : l'écart restant vient de la qualité des solutions.
            el, tk = charger_eleve(cfg)
            ta = cfg.temoin_auto
            propres = modeles.generer(el, tk, [modeles.prompt_texte(tk, q["question"], amorce) for q in questions],
                                      max_new_tokens=cfg.evaluation.max_new_tokens, temperature=ta.temperature, n=ta.k,
                                      batch=cfg.generation.batch // ta.k or 1)
            del el
            torch.cuda.empty_cache()
            auto = []
            for q, rs in zip(questions, propres):
                justes = [r for r in rs if not cfg.generation.filtrer or evaluation.correct(r, q["cible"], cfg.evaluation.mode)]
                if justes:
                    auto.append({"question": q["question"], "reponse": justes[0].strip(), "cible": q["cible"]})
            random.Random(cfg.graine).shuffle(auto)
            auto = auto[: len(corpus)]
            print(f"[auto-distillation] {len(auto)} solutions de l'élève gardées (enseignant : {len(corpus)})")
            res = entrainer_temoin("temoin-auto", auto)
            temoin |= {"temoin_auto": res,
                       "distille_contre_auto": preuve.difference_appariee(res["tache"]["succes"], apres["tache"]["succes"]),
                       "auto_contre_avant": preuve.difference_appariee(avant["tache"]["succes"], res["tache"]["succes"])}
    rapport.enregistrer(cfg, avant=avant, apres=apres, **temoin, enseignant=ref_enseignant, mode=cfg.mode,
                        **({"reponses_conservees": len(corpus)} if cfg.mode == "sequence" else {}))


if __name__ == "__main__":
    main()
