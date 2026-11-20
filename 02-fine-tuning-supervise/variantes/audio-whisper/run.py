"""Fine-tuning supervisé d'un modèle de reconnaissance vocale (Whisper) sur une langue ou un domaine.

Cas typique : accent, vocabulaire métier (médical, noms propres), langue mal servie. Mesure : WER (part de mots
faux), sur le texte normalisé par le normaliseur anglais de Whisper quand la langue est l'anglais : « eleven » et
« 11 », majuscules et ponctuation ne comptent pas comme des erreurs. L'écart avant/après porte sur les mêmes
segments (bootstrap apparié). `non_regression` : un autre jeu de parole, qui ne doit pas se dégrader.
"""

import pathlib
import re
import sys

sys.path.insert(0, str(next(p for p in pathlib.Path(__file__).resolve().parents if (p / "commun").is_dir())))

import numpy as np
import torch
from datasets import Audio, load_dataset
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, WhisperForConditionalGeneration, WhisperProcessor

from commun import config, entrainement, modeles, rapport, reprise


def normaliser(t: str, normaliseur=None) -> list[str]:
    if normaliseur:
        t = normaliseur(t)
    return re.sub(r"[^\w\s']", " ", t.lower()).split()


def erreurs_par_segment(references: list[str], hypotheses: list[str], normaliseur=None) -> tuple[list[int], list[int]]:
    """Distance d'édition en mots et nombre de mots de référence, segment par segment."""
    erreurs, longueurs = [], []
    for ref, hyp in zip(references, hypotheses):
        r, h = normaliser(ref, normaliseur), normaliser(hyp, normaliseur)
        dist = np.arange(len(h) + 1)
        for i in range(1, len(r) + 1):
            prec, dist[0] = dist.copy(), i
            for j in range(1, len(h) + 1):
                dist[j] = min(prec[j] + 1, dist[j - 1] + 1, prec[j - 1] + (r[i - 1] != h[j - 1]))
        erreurs.append(int(dist[len(h)]))
        longueurs.append(len(r))
    return erreurs, longueurs


def wer(references: list[str], hypotheses: list[str], normaliseur=None) -> float:
    e, n = erreurs_par_segment(references, hypotheses, normaliseur)
    return round(sum(e) / max(sum(n), 1), 4)


def ecart_wer(avant: dict, apres: dict, tirages: int = 10_000, graine: int = 0) -> dict:
    """WER après − WER avant sur les mêmes segments, intervalle à 95 % par bootstrap apparié des segments."""
    ea, eb, n = (np.array(x) for x in (avant["erreurs"], apres["erreurs"], avant["longueurs"]))
    rng = np.random.default_rng(graine)
    idx = rng.integers(0, len(n), (tirages, len(n)))
    diffs = np.sort((eb[idx].sum(1) - ea[idx].sum(1)) / np.maximum(n[idx].sum(1), 1))
    bas, haut = float(diffs[int(0.025 * tirages)]), float(diffs[int(0.975 * tirages) - 1])
    return {"difference": round((eb.sum() - ea.sum()) / max(n.sum(), 1), 4), "ic95": [round(bas, 4), round(haut, 4)],
            "n_segments": int(len(n)), "net": bas > 0 or haut < 0}


def main():
    cfg = config.charger(__file__)
    d = cfg.donnees
    processeur = WhisperProcessor.from_pretrained(cfg.modele.nom, language=cfg.modele.get("langue"), task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(cfg.modele.nom).to(modeles.appareil())
    model.generation_config.language = cfg.modele.get("langue")
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None

    normaliseur = processeur.tokenizer.normalize if cfg.modele.get("langue") == "english" else None

    def utilisable(texte, audio):
        # segments transcrits d'au moins `min_mots` mots et de 30 s au plus (fenêtre de Whisper)
        return len(texte.split()) >= d.get("min_mots", 1) and len(audio["array"]) <= 30 * 16000

    def charger(split, src=d):
        ds_ = load_dataset(src.nom, src.get("config"), split=split).cast_column(src.colonne_audio, Audio(sampling_rate=16000))
        if src.get("min_mots"):
            ds_ = ds_.filter(utilisable, input_columns=[src.colonne_texte, src.colonne_audio])
        return ds_.shuffle(seed=cfg.graine)

    ds = charger(d.split)
    if d.get("split_test"):   # split de test séparé (locuteurs distincts)
        test_complet = charger(d.split_test)
        test = test_complet.select(range(min(d.n_test, len(test_complet))))
        train = ds.select(range(min(d.n_train, len(ds))))
    else:                     # un seul split : les n_test premiers exemples servent au test
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

    nr = cfg.get("non_regression")
    test_nr = None
    if nr:
        test_nr = load_dataset(nr.nom, nr.get("config"), split=nr.split).cast_column(nr.colonne_audio, Audio(sampling_rate=16000))
        test_nr = test_nr.shuffle(seed=cfg.graine).select(range(min(nr.n, len(test_nr))))

    @torch.no_grad()
    def transcrire(m, jeu, colonne_audio):
        m.eval()
        hyps = []
        for i in range(0, len(jeu), 16):
            lot = jeu.select(range(i, min(i + 16, len(jeu))))
            feats = processeur.feature_extractor([a["array"] for a in lot[colonne_audio]], sampling_rate=16000, return_tensors="pt")
            out = m.generate(feats.input_features.to(m.device, m.dtype), max_new_tokens=128)
            hyps += processeur.batch_decode(out, skip_special_tokens=True)
        return hyps

    def evaluer(m):
        hyps = transcrire(m, test, d.colonne_audio)
        refs = test[d.colonne_texte]
        erreurs, longueurs = erreurs_par_segment(refs, hyps, normaliseur)
        res = {"wer": round(sum(erreurs) / max(sum(longueurs), 1), 4), "erreurs": erreurs, "longueurs": longueurs,
               "exemples": [{"reference": r, "transcription": h} for r, h in list(zip(refs, hyps))[:3]]}
        if test_nr is not None:
            res["non_regression_wer"] = wer(test_nr[nr.colonne_texte], transcrire(m, test_nr, nr.colonne_audio), normaliseur)
        return res

    avant = evaluer(model)
    trainer = Seq2SeqTrainer(
        model=model,
        args=Seq2SeqTrainingArguments(**entrainement.arguments(cfg.entrainement, cfg.sortie, remove_unused_columns=False)),
        train_dataset=train_enc, data_collator=collate,
    )
    reprise.entrainer(trainer)
    apres = evaluer(trainer.model)

    if cfg.get("sauvegarder", True):
        trainer.save_model(f"{cfg.sortie}/modele")
        processeur.save_pretrained(f"{cfg.sortie}/modele")
    rapport.enregistrer(cfg, avant=avant, apres=apres, wer_apres_moins_avant=ecart_wer(avant, apres))


if __name__ == "__main__":
    main()
