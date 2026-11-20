"""Fine-tuning supervisé d'un modèle ENCODEUR-DÉCODEUR (T5, FLAN-T5, BART, mT5, NLLB...).

Même format question -> réponse que le SFT de LLM, mais l'entrée passe par l'encodeur et la réponse est produite
par le décodeur (traduction, résumé, reformulation, extraction...).

Tâche par défaut : résumer des conversations (DialogSum), mesurée en ROUGE-L. Chaque conversation de test a
plusieurs résumés de référence ; un résumé produit est noté sur la référence dont il est le plus proche. Avant et
après sont comparés conversation par conversation (écart moyen apparié).
"""

import collections
import pathlib
import re
import sys

sys.path.insert(0, str(next(p for p in pathlib.Path(__file__).resolve().parents if (p / "commun").is_dir())))

import torch
from datasets import Dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments

from commun import config, donnees, entrainement, evaluation, modeles, preuve, rapport, reprise


def mots(texte):
    return re.findall(r"\w+", texte.lower())


def rouge_l(produit, reference):
    """ROUGE-L F1 : plus longue sous-suite commune de mots, rapportée aux longueurs des deux textes."""
    a, b = mots(produit), mots(reference)
    if not a or not b:
        return 0.0
    precedent = [0] * (len(b) + 1)
    for x in a:
        courant = [0]
        for j, y in enumerate(b):
            courant.append(precedent[j] + 1 if x == y else max(precedent[j + 1], courant[j]))
        precedent = courant
    lcs = precedent[-1]
    if lcs == 0:
        return 0.0
    p, r = lcs / len(a), lcs / len(b)
    return 2 * p * r / (p + r)


def references_groupees(exemples):
    """Une entrée par question, avec toutes ses réponses de référence (jeux à plusieurs références)."""
    groupes = collections.OrderedDict()
    for ex in exemples:
        g = groupes.setdefault(ex["question"], {**ex, "references": [], "cibles": []})
        g["references"].append(ex["reponse"])
        g["cibles"].append(ex["cible"])
    return list(groupes.values())


@torch.no_grad()
def mesurer(model, tok, exemples, cfg):
    model.eval()
    ev, sorties = cfg.evaluation, []
    for i in range(0, len(exemples), ev.batch):
        lot = exemples[i : i + ev.batch]
        enc = tok([cfg.donnees.consigne + ex["question"] for ex in lot], return_tensors="pt", padding=True,
                  truncation=True, max_length=cfg.entrainement.max_length).to(model.device)
        out = model.generate(**enc, max_new_tokens=ev.max_new_tokens, num_beams=ev.get("num_beams", 1))
        sorties += tok.batch_decode(out, skip_special_tokens=True)
    exemple = {"question": exemples[0]["question"][:300], "sortie": sorties[0], "attendu": exemples[0]["references"][0]}
    if ev.mode == "rouge_l":
        scores = [max(rouge_l(s, r) for r in ex["references"]) for s, ex in zip(sorties, exemples)]
        return {"rouge_l": round(sum(scores) / len(scores), 4), "n": len(scores),
                "mots_par_resume": round(sum(len(mots(s)) for s in sorties) / len(sorties), 1),
                "exemple": exemple, "par_exemple": scores}
    ok = [any(evaluation.correct(s, c, ev.mode) for c in ex["cibles"]) for s, ex in zip(sorties, exemples)]
    return {**preuve.taux(ok), "exemple": exemple}


def main():
    cfg = config.charger(__file__)
    tok = AutoTokenizer.from_pretrained(cfg.modele.nom)
    model = AutoModelForSeq2SeqLM.from_pretrained(cfg.modele.nom).to(modeles.appareil())
    train = donnees.charger_qa(cfg.donnees, "train")
    test = references_groupees(donnees.charger_qa(cfg.donnees, "test", cfg.donnees.get("n_test_lignes")))[: cfg.evaluation.n]

    def encoder(ex):
        enc = tok(cfg.donnees.consigne + ex["question"], truncation=True, max_length=cfg.entrainement.max_length)
        enc["labels"] = tok(text_target=ex["reponse"], truncation=True, max_length=cfg.entrainement.max_length_cible)["input_ids"]
        return enc

    ds = Dataset.from_list(train).map(encoder, remove_columns=list(train[0].keys()))
    avant = mesurer(model, tok, test, cfg)
    trainer = Seq2SeqTrainer(
        model=model,
        args=Seq2SeqTrainingArguments(**entrainement.arguments(cfg.entrainement, cfg.sortie, bf16=False)),
        train_dataset=ds, data_collator=DataCollatorForSeq2Seq(tok, model=model),
    )
    with entrainement.mesurer() as mesures:
        reprise.entrainer(trainer)
    apres = mesurer(trainer.model, tok, test, cfg)

    if cfg.get("sauvegarder", True):
        trainer.save_model(f"{cfg.sortie}/modele")
        tok.save_pretrained(f"{cfg.sortie}/modele")
    ecart = ({"rouge_l_apres_moins_avant": preuve.ecart_moyen_apparie(avant.pop("par_exemple"), apres.pop("par_exemple"))}
             if "par_exemple" in avant else {})
    rapport.enregistrer(cfg, avant=avant, apres=apres, **ecart, entrainement=dict(mesures))


if __name__ == "__main__":
    main()
