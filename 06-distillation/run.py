"""Distillation : un gros modèle « enseignant » transmet son comportement à un petit modèle « élève ».

Trois modes (`--set mode=...`), du plus simple au plus fidèle :
- sequence : l'enseignant rédige les réponses, l'élève fait un SFT dessus (pas besoin du même tokenizer).
- logits   : l'élève imite la distribution complète de l'enseignant token par token (KL), sur des réponses fixes.
- on_policy: l'élève génère, l'enseignant corrige sa distribution sur les propres réponses de l'élève (GKD).
Étapes : évaluer enseignant et élève -> distiller -> réévaluer l'élève -> sauvegarder.
"""

import gc
import pathlib
import sys

sys.path.insert(0, str(next(p for p in pathlib.Path(__file__).resolve().parents if (p / "commun").is_dir())))

import torch
import torch.nn.functional as F
from datasets import Dataset
from transformers import Trainer, TrainingArguments

from commun import config, donnees, entrainement, evaluation, modeles, preuve, rapport


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


def main():
    cfg = config.charger(__file__)
    eleve, tok = modeles.charger_llm(cfg.eleve.nom, cfg.eleve.get("dtype", "auto"))
    enseignant, tok_ens = modeles.charger_llm(cfg.enseignant.nom)
    enseignant.requires_grad_(False)

    questions = donnees.charger_qa(cfg.donnees, "train")
    test = donnees.charger_qa(cfg.donnees, "test", cfg.evaluation.n)
    ref_enseignant = evaluation.evaluer(enseignant, tok_ens, cfg, test)
    avant = evaluation.evaluer(eleve, tok, cfg, test)
    e = cfg.entrainement

    if cfg.mode == "sequence":
        # L'enseignant rédige ; on ne garde que les réponses justes quand on peut vérifier (distillation filtrée).
        sorties = modeles.generer(enseignant, tok_ens, [modeles.prompt_texte(tok_ens, q["question"]) for q in questions],
                                  max_new_tokens=cfg.evaluation.max_new_tokens, batch=cfg.generation.batch)
        corpus = [{"question": q["question"], "reponse": s.strip(), "cible": q["cible"]} for q, s in zip(questions, sorties)
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
        trainer.train()
    elif cfg.mode == "on_policy":
        from trl import DistillationConfig, DistillationTrainer

        ds = Dataset.from_list([{"prompt": modeles.messages(q["question"])} for q in questions])
        args = DistillationConfig(**entrainement.arguments(e, cfg.sortie, max_completion_length=cfg.evaluation.max_new_tokens,
                                                           temperature=cfg.kd.temperature, beta=cfg.kd.beta_jsd))
        trainer = DistillationTrainer(model=eleve, teacher_model=enseignant, args=args, train_dataset=ds, processing_class=tok)
        trainer.train()
    else:
        raise ValueError(cfg.mode)

    apres = evaluation.evaluer(trainer.model, tok, cfg, test)
    if cfg.get("sauvegarder", True):
        trainer.save_model(f"{cfg.sortie}/eleve-{cfg.mode}")
        tok.save_pretrained(f"{cfg.sortie}/eleve-{cfg.mode}")
    temoin = {}
    if cfg.mode == "sequence" and cfg.get("temoin"):
        # Témoin : le même élève, entraîné de la même façon sur les mêmes questions, mais avec les solutions écrites
        # par les auteurs du jeu au lieu de celles de l'enseignant. L'écart mesure ce qu'apporte l'enseignant.
        entrainement.liberer(trainer)
        del trainer, eleve
        gc.collect()
        torch.cuda.empty_cache()
        eleve, tok = modeles.charger_llm(cfg.eleve.nom, cfg.eleve.get("dtype", "auto"))
        retenues = {c["question"] for c in corpus}
        humaines = [q for q in questions if q["question"] in retenues]
        trainer = entrainement.sft(eleve, tok, entrainement.exemples_sft(tok, humaines), e, f"{cfg.sortie}/temoin",
                                   max_length=e.max_length)
        res_temoin = evaluation.evaluer(trainer.model, tok, cfg, test)
        temoin = {"temoin": res_temoin, "distille_contre_temoin": preuve.difference_appariee(
            res_temoin["tache"]["succes"], apres["tache"]["succes"])}
    rapport.enregistrer(cfg, avant=avant, apres=apres, **temoin, enseignant=ref_enseignant, mode=cfg.mode,
                        **({"reponses_conservees": len(corpus)} if cfg.mode == "sequence" else {}))


if __name__ == "__main__":
    main()
