"""Fine-tuning supervisé d'un modèle de reconnaissance vocale (Whisper) sur une langue ou un domaine.

Cas typique : accent, vocabulaire métier (médical, noms propres), langue mal servie. Mesure : WER.
"""

import pathlib
import re
import sys

sys.path.insert(0, str(next(p for p in pathlib.Path(__file__).resolve().parents if (p / "commun").is_dir())))

import numpy as np
import torch
from datasets import Audio, load_dataset
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, WhisperForConditionalGeneration, WhisperProcessor

from commun import config, entrainement, modeles, rapport


def normaliser(t: str) -> list[str]:
    return re.sub(r"[^\w\s']", " ", t.lower()).split()


def wer(references: list[str], hypotheses: list[str]) -> float:
    erreurs, total = 0, 0
    for ref, hyp in zip(references, hypotheses):
        r, h = normaliser(ref), normaliser(hyp)
        dist = np.arange(len(h) + 1)
        for i in range(1, len(r) + 1):
            prec, dist[0] = dist.copy(), i
            for j in range(1, len(h) + 1):
                dist[j] = min(prec[j] + 1, dist[j - 1] + 1, prec[j - 1] + (r[i - 1] != h[j - 1]))
        erreurs += dist[len(h)]
        total += len(r)
    return round(erreurs / max(total, 1), 4)


def main():
    cfg = config.charger(__file__)
    d = cfg.donnees
    processeur = WhisperProcessor.from_pretrained(cfg.modele.nom, language=cfg.modele.get("langue"), task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(cfg.modele.nom).to(modeles.appareil())
    model.generation_config.language = cfg.modele.get("langue")
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None

    ds = load_dataset(d.nom, d.get("config"), split=d.split).cast_column(d.colonne_audio, Audio(sampling_rate=16000))
    ds = ds.shuffle(seed=cfg.graine)
    test = ds.select(range(d.n_test))
    train = ds.select(range(d.n_test, min(d.n_test + d.n_train, len(ds))))

    def preparer(ex):
        audio = ex[d.colonne_audio]
        tableau = audio["array"]
        return {
            "input_features": processeur.feature_extractor(tableau, sampling_rate=16000).input_features[0],
            "labels": processeur.tokenizer(ex[d.colonne_texte]).input_ids,
        }

    train_enc = train.map(preparer, remove_columns=train.column_names)

    def collate(lot):
        feats = processeur.feature_extractor.pad([{"input_features": x["input_features"]} for x in lot], return_tensors="pt")
        labels = processeur.tokenizer.pad([{"input_ids": x["labels"]} for x in lot], return_tensors="pt")
        ids = labels["input_ids"].masked_fill(labels["attention_mask"] == 0, -100)
        if (ids[:, 0] == model.config.decoder_start_token_id).all():
            ids = ids[:, 1:]
        return {"input_features": feats["input_features"], "labels": ids}

    @torch.no_grad()
    def evaluer(m):
        m.eval()
        hyps = []
        for i in range(0, len(test), 16):
            lot = test.select(range(i, min(i + 16, len(test))))
            feats = processeur.feature_extractor([a["array"] for a in lot[d.colonne_audio]], sampling_rate=16000, return_tensors="pt")
            out = m.generate(feats.input_features.to(m.device, m.dtype), max_new_tokens=128)
            hyps += processeur.batch_decode(out, skip_special_tokens=True)
        return {"wer": wer(test[d.colonne_texte], hyps), "exemple": hyps[0]}

    avant = evaluer(model)
    trainer = Seq2SeqTrainer(
        model=model,
        args=Seq2SeqTrainingArguments(**entrainement.arguments(cfg.entrainement, cfg.sortie, remove_unused_columns=False)),
        train_dataset=train_enc, data_collator=collate,
    )
    trainer.train()
    apres = evaluer(trainer.model)

    if cfg.get("sauvegarder", True):
        trainer.save_model(f"{cfg.sortie}/modele")
        processeur.save_pretrained(f"{cfg.sortie}/modele")
    rapport.enregistrer(cfg, avant=avant, apres=apres)


if __name__ == "__main__":
    main()
