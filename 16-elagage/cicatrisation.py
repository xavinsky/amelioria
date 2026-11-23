"""Cicatrisation par distillation sur les sorties pré-calculées de l'enseignant, avec un LoRA non fusionné.

L'enseignant (le modèle d'origine, dense) passe une fois sur les blocs de texte avant l'élagage : on garde, à chaque
position, ses `k` tokens les plus probables et leurs log-probabilités. Il n'est donc plus en mémoire pendant
l'entraînement, ce qui laisse la place à un élève plus gros. L'élève élagué reste creux et gelé, seul un adaptateur
LoRA apprend ; il n'est pas fusionné, pour que les poids nuls le restent.
"""

import pathlib

import torch
import torch.nn.functional as F
from transformers import Trainer, TrainingArguments, default_data_collator

from commun import entrainement, reprise


class BlocsEnseignant(torch.utils.data.Dataset):
    def __init__(self, ids, top_idx, top_lp):
        self.ids, self.top_idx, self.top_lp = ids, top_idx, top_lp

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        return {"input_ids": self.ids[i], "top_idx": self.top_idx[i], "top_lp": self.top_lp[i]}


@torch.no_grad()
def precalculer(model, blocs, k: int, fichier: pathlib.Path, batch: int = 4) -> BlocsEnseignant:
    """Sorties de `model` sur les blocs (Dataset `input_ids`) : indices et log-probabilités des `k` premiers tokens par position."""
    if fichier.exists():
        d = torch.load(fichier)
        print(f"[cicatrisation] sorties de l'enseignant relues : {fichier.name}")
        return BlocsEnseignant(**d)
    model.eval()
    dev = next(model.parameters()).device
    ids = torch.tensor(blocs["input_ids"])
    idx, lp = [], []
    for i in range(0, len(ids), batch):
        logits = model(input_ids=ids[i : i + batch].to(dev)).logits.float()
        haut = F.log_softmax(logits, -1).topk(k, dim=-1)
        idx.append(haut.indices.int().cpu())
        lp.append(haut.values.half().cpu())
    d = {"ids": ids, "top_idx": torch.cat(idx), "top_lp": torch.cat(lp)}
    fichier.parent.mkdir(parents=True, exist_ok=True)
    torch.save(d, fichier)
    print(f"[cicatrisation] sorties de l'enseignant calculées : {len(ids)} blocs, top-{k}, {fichier.stat().st_size / 2**20:.0f} Mo")
    return BlocsEnseignant(**d)


class TrainerTopK(Trainer):
    """KL entre l'enseignant et l'élève, restreinte aux `k` tokens de l'enseignant et renormalisée sur eux."""

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        sortie = model(input_ids=inputs["input_ids"])
        lp_e = sortie.logits.float().gather(-1, inputs["top_idx"].long())
        lp_e = lp_e - torch.logsumexp(lp_e, -1, keepdim=True)
        lp_t = inputs["top_lp"].float()
        lp_t = lp_t - torch.logsumexp(lp_t, -1, keepdim=True)
        perte = (lp_t.exp() * (lp_t - lp_e)).sum(-1).mean()
        return (perte, sortie) if return_outputs else perte


def cicatriser(model, donnees_enseignant: BlocsEnseignant, lora, e, sortie: str, callbacks=None):
    """Ajoute l'adaptateur LoRA (non fusionné) à `model` élagué et l'entraîne ; renvoie le modèle avec adaptateur."""
    from peft import get_peft_model

    if e.get("gradient_checkpointing"):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(model, entrainement.config_lora(lora))
    model.print_trainable_parameters()
    trainer = TrainerTopK(model=model, args=TrainingArguments(**entrainement.arguments(e, sortie, remove_unused_columns=False)),
                          train_dataset=donnees_enseignant, data_collator=default_data_collator, callbacks=callbacks)
    reprise.entrainer(trainer)
    return entrainement.liberer(trainer).model
