"""Distillation de séquences à grande échelle : l'enseignant (70B, DeepSeek-R1...) génère avec vLLM.

Sortie : JSONL {question, reponse, cible}, directement utilisable par le SFT du dossier 02 :
    --set donnees.source=hf --set donnees.nom=json --set donnees.fichiers=<sortie.jsonl>
    --set donnees.champs='{question: question, reponse: reponse, cible: cible}' --set donnees.split_train=train

    pip install vllm
    python 06-distillation/infra/generer_vllm.py --enseignant deepseek-ai/DeepSeek-R1-Distill-Llama-70B --tp 8 \
        --dataset openai/gsm8k --config main --colonne question --cible answer --n 16 --sortie distill.jsonl
"""

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(next(p for p in pathlib.Path(__file__).resolve().parents if (p / "commun").is_dir())))

from commun import donnees  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--enseignant", required=True)
    p.add_argument("--tp", type=int, default=8, help="tensor parallel")
    p.add_argument("--dataset", required=True)
    p.add_argument("--config", default=None)
    p.add_argument("--split", default="train")
    p.add_argument("--colonne", default="question")
    p.add_argument("--cible", default=None, help="colonne de réponse de référence, pour filtrer les réponses fausses")
    p.add_argument("--n", type=int, default=4, help="réponses par question (rejection sampling si --cible)")
    p.add_argument("--max-tokens", type=int, default=8192)
    p.add_argument("--sortie", required=True)
    a = p.parse_args()

    from datasets import load_dataset
    from vllm import LLM, SamplingParams

    ds = load_dataset(a.dataset, a.config, split=a.split)
    llm = LLM(a.enseignant, tensor_parallel_size=a.tp, max_model_len=a.max_tokens + 2048)
    messages = [[{"role": "user", "content": q}] for q in ds[a.colonne]]
    sorties = llm.chat(messages, SamplingParams(n=a.n, temperature=0.7, top_p=0.95, max_tokens=a.max_tokens))

    gardees = 0
    with open(a.sortie, "w", encoding="utf-8") as f:
        for ex, out in zip(ds, sorties):
            cible = None
            if a.cible:
                cible = str(ex[a.cible]).rpartition("####")[2].strip().replace(",", "")
            for c in out.outputs:
                if cible is None or donnees.extraire_nombre(c.text) == cible:
                    f.write(json.dumps({"question": ex[a.colonne], "reponse": c.text, "cible": cible or ""}, ensure_ascii=False) + "\n")
                    gardees += 1
                    break   # une réponse juste par question suffit (augmenter pour plus de diversité)
    print(f"{gardees}/{len(ds)} questions avec une réponse conservée -> {a.sortie}")


if __name__ == "__main__":
    main()
