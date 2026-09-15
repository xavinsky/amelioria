"""Quantification : représenter les poids (et parfois les activations) sur moins de bits pour un modèle plus petit et plus rapide.

Méthodes (`--set methode=...`) :
- sans calibration (PTQ « à la volée ») : bnb_nf4, bnb_int8 (bitsandbytes), torchao_int8wo, torchao_int4wo, torchao_fp8
- avec calibration (PTQ) : gptq, awq (llm-compressor, format compressed-tensors lisible par vLLM)
- pendant l'entraînement : qat (torchao, fake-quant pendant un court fine-tuning puis conversion int4)
Étapes : mesurer le modèle bf16 -> quantifier -> mesurer (mémoire, perplexité, exactitude, vitesse) -> sauvegarder.
"""

import pathlib
import sys
import time

sys.path.insert(0, str(next(p for p in pathlib.Path(__file__).resolve().parents if (p / "commun").is_dir())))

import torch
from transformers import AutoModelForCausalLM

from commun import config, donnees, entrainement, evaluation, modeles, rapport


def mesurer(model, tok, cfg, test, textes):
    dev = next(model.parameters()).device
    prompt = tok(modeles.prompt_texte(tok, "Write a short paragraph about the ocean."), return_tensors="pt").to(dev)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    out = model.generate(**prompt, max_new_tokens=64, do_sample=False, min_new_tokens=64)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    duree = time.time() - t0
    res = {
        "memoire_modele_Mo": round(model.get_memory_footprint() / 2**20, 1),
        "tokens_par_seconde": round((out.shape[1] - prompt["input_ids"].shape[1]) / duree, 1),
        "perplexite_generale": evaluation.perplexite(model, tok, textes, 256),
    }
    res.update(evaluation.evaluer(model, tok, cfg, test))
    return res


def charger_quantifie(cfg, tok):
    """Renvoie (modèle quantifié, dossier sauvegardé ou None)."""
    m, nom = cfg.methode, cfg.modele.nom
    dtype = modeles.dtype_calcul()
    if m in ("bnb_nf4", "bnb_int8"):
        return modeles.charger_llm(nom, quantification="4bit" if m == "bnb_nf4" else "8bit")[0], None
    if m.startswith("torchao"):
        from torchao.quantization import Float8DynamicActivationFloat8WeightConfig, Int4WeightOnlyConfig, Int8WeightOnlyConfig
        from transformers import TorchAoConfig

        schema = {"torchao_int8wo": Int8WeightOnlyConfig(), "torchao_int4wo": Int4WeightOnlyConfig(group_size=128),
                  "torchao_fp8": Float8DynamicActivationFloat8WeightConfig()}[m]
        model = AutoModelForCausalLM.from_pretrained(nom, dtype=dtype, device_map=str(modeles.appareil()),
                                                     quantization_config=TorchAoConfig(quant_type=schema))
        return model, None
    if m in ("gptq", "awq"):
        from llmcompressor import oneshot

        model = AutoModelForCausalLM.from_pretrained(nom, dtype=dtype)
        if m == "gptq":
            from llmcompressor.modifiers.quantization import GPTQModifier

            recette = GPTQModifier(targets="Linear", scheme=cfg.gptq.schema, ignore=["lm_head"])
        else:
            from llmcompressor.modifiers.awq import AWQModifier

            recette = AWQModifier(targets=["Linear"], scheme="W4A16_ASYM", ignore=["lm_head"])
        calibration = [{"text": t} for t in donnees.textes_generaux(cfg.calibration.n, graine=7)]
        from datasets import Dataset

        ds = Dataset.from_list(calibration).map(lambda x: tok(x["text"], truncation=True, max_length=cfg.calibration.longueur),
                                                remove_columns=["text"])
        oneshot(model=model, dataset=ds, recipe=recette, max_seq_length=cfg.calibration.longueur, num_calibration_samples=cfg.calibration.n)
        dossier = f"{cfg.sortie}/modele-{m}"
        model.save_pretrained(dossier, save_compressed=True)
        tok.save_pretrained(dossier)
        return AutoModelForCausalLM.from_pretrained(dossier, device_map=str(modeles.appareil())), dossier
    if m == "qat":
        from torchao.quantization import Int4WeightOnlyConfig, quantize_
        from torchao.quantization.qat import QATConfig

        model, _ = modeles.charger_llm(nom, "float32")
        base = Int4WeightOnlyConfig(group_size=cfg.qat.group_size)
        quantize_(model, QATConfig(base, step="prepare"))   # fausse quantification : le modèle apprend à tolérer l'int4
        qa = donnees.charger_qa(cfg.donnees, "train", cfg.qat.n_train)
        trainer = entrainement.sft(model, tok, entrainement.exemples_sft(tok, qa), cfg.qat.entrainement, f"{cfg.sortie}/qat")
        model = trainer.model.to(dtype)
        quantize_(model, QATConfig(base, step="convert"))   # vraie quantification int4
        return model, None
    raise ValueError(f"méthode inconnue : {m}")


def main():
    cfg = config.charger(__file__)
    tok = modeles.charger_tokenizer(cfg.modele.nom)
    test = donnees.charger_qa(cfg.donnees, "test", cfg.evaluation.n)
    textes = donnees.textes_generaux(cfg.evaluation.n_textes_generaux, graine=123)

    reference, _ = modeles.charger_llm(cfg.modele.nom)
    res = {"reference_bf16": mesurer(reference, tok, cfg, test, textes)}
    del reference
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model, dossier = charger_quantifie(cfg, tok)
    res[cfg.methode] = mesurer(model, tok, cfg, test, textes)
    if cfg.get("sauvegarder", True) and dossier is None and cfg.methode.startswith(("bnb", "torchao")):
        model.save_pretrained(f"{cfg.sortie}/modele-{cfg.methode}", safe_serialization=not cfg.methode.startswith("torchao"))
        tok.save_pretrained(f"{cfg.sortie}/modele-{cfg.methode}")
    rapport.enregistrer(cfg, **res)


if __name__ == "__main__":
    main()
