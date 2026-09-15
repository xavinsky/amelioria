"""Quantification d'un 70B+ avec llm-compressor, sans charger le modèle entier sur un GPU :
les couches sont calibrées une par une (onloading séquentiel). Sortie au format compressed-tensors, servie par vLLM.

    pip install llmcompressor
    python 15-quantification/infra/gptq_70b_llmcompressor.py --modele meta-llama/Llama-3.3-70B-Instruct --schema W4A16
    vllm serve sorties/Llama-3.3-70B-Instruct-W4A16 --tensor-parallel-size 2
Schémas : W4A16 (GPTQ int4, mémoire ÷4) ; W8A8 (int8 poids+activations, débit x1,5-2) ; FP8_DYNAMIC (sans calibration, H100/L40S+).
"""

import argparse


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--modele", required=True)
    p.add_argument("--schema", default="W4A16")
    p.add_argument("--n", type=int, default=512)
    p.add_argument("--longueur", type=int, default=2048)
    a = p.parse_args()

    from datasets import load_dataset
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import GPTQModifier, QuantizationModifier
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(a.modele, dtype="auto")
    tok = AutoTokenizer.from_pretrained(a.modele)
    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split=f"train_sft[:{a.n}]")
    ds = ds.map(lambda x: tok(tok.apply_chat_template(x["messages"], tokenize=False), truncation=True, max_length=a.longueur),
                remove_columns=ds.column_names)

    if a.schema == "FP8_DYNAMIC":
        recette = QuantizationModifier(targets="Linear", scheme="FP8_DYNAMIC", ignore=["lm_head"])
        oneshot(model=model, recipe=recette)
    else:
        recette = GPTQModifier(targets="Linear", scheme=a.schema, ignore=["lm_head"])
        oneshot(model=model, dataset=ds, recipe=recette, max_seq_length=a.longueur, num_calibration_samples=a.n)

    dossier = f"sorties/{a.modele.split('/')[-1]}-{a.schema}"
    model.save_pretrained(dossier, save_compressed=True)
    tok.save_pretrained(dossier)
    print(f"-> {dossier}")


if __name__ == "__main__":
    main()
