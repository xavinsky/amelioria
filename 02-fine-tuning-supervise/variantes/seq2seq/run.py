"""Fine-tuning supervisé d'un modèle ENCODEUR-DÉCODEUR (T5, FLAN-T5, BART, mT5, NLLB...).

Même données question -> réponse que le SFT de LLM, mais l'entrée passe par l'encodeur et la réponse
est produite par le décodeur (traduction, résumé, reformulation, extraction...).
"""

import pathlib
import sys

sys.path.insert(0, str(next(p for p in pathlib.Path(__file__).resolve().parents if (p / "commun").is_dir())))

import torch
from datasets import Dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments

from commun import config, donnees, entrainement, evaluation, modeles, rapport


@torch.no_grad()
def exactitude(model, tok, exemples, cfg):
    model.eval()
    ok, sorties = 0, []
    for i in range(0, len(exemples), 32):
        lot = exemples[i : i + 32]
        enc = tok([ex["question"] for ex in lot], return_tensors="pt", padding=True).to(model.device)
        out = model.generate(**enc, max_new_tokens=cfg.evaluation.max_new_tokens)
        textes = tok.batch_decode(out, skip_special_tokens=True)
        sorties += textes
        ok += sum(evaluation.correct(t, ex["cible"], cfg.evaluation.mode) for t, ex in zip(textes, lot))
    return {"exactitude": round(ok / len(exemples), 4), "exemple": sorties[0]}


def main():
    cfg = config.charger(__file__)
    tok = AutoTokenizer.from_pretrained(cfg.modele.nom)
    model = AutoModelForSeq2SeqLM.from_pretrained(cfg.modele.nom).to(modeles.appareil())
    train = donnees.charger_qa(cfg.donnees, "train")
    test = donnees.charger_qa(cfg.donnees, "test", cfg.evaluation.n)

    def encoder(ex):
        enc = tok(ex["question"], truncation=True, max_length=cfg.entrainement.max_length)
        enc["labels"] = tok(text_target=ex["reponse"], truncation=True, max_length=cfg.entrainement.max_length)["input_ids"]
        return enc

    ds = Dataset.from_list(train).map(encoder, remove_columns=["question", "reponse", "cible"])
    avant = exactitude(model, tok, test, cfg)
    trainer = Seq2SeqTrainer(
        model=model,
        args=Seq2SeqTrainingArguments(**entrainement.arguments(cfg.entrainement, cfg.sortie, bf16=False)),
        train_dataset=ds, data_collator=DataCollatorForSeq2Seq(tok, model=model),
    )
    trainer.train()
    apres = exactitude(trainer.model, tok, test, cfg)

    if cfg.get("sauvegarder", True):
        trainer.save_model(f"{cfg.sortie}/modele")
        tok.save_pretrained(f"{cfg.sortie}/modele")
    rapport.enregistrer(cfg, avant=avant, apres=apres)


if __name__ == "__main__":
    main()
