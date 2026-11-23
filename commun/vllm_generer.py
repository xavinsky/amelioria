"""Génération par vLLM, exécutée dans le venv de vLLM (processus séparé de celui du dépôt).

    python commun/vllm_generer.py demande.json reponse.json

`demande.json` : modele, prompts, max_new_tokens, temperature, top_p, n, stop, gpu_memory_utilization, max_model_len.
`reponse.json` : une liste de n textes par prompt, chaque suite s'arrêtant d'elle-même (stop ou fin de séquence).
"""

import json
import sys


def main(chemin_demande: str, chemin_reponse: str):
    from vllm import LLM, SamplingParams

    d = json.load(open(chemin_demande, encoding="utf-8"))
    llm = LLM(d["modele"], dtype=d.get("dtype", "bfloat16"), gpu_memory_utilization=d.get("gpu_memory_utilization", 0.6),
              max_model_len=d.get("max_model_len", 4096), enable_prefix_caching=True, seed=d.get("graine", 0))
    params = SamplingParams(n=d.get("n", 1), temperature=d.get("temperature", 0.0), top_p=d.get("top_p", 1.0),
                            max_tokens=d["max_new_tokens"], stop=d.get("stop") or None)
    sorties = llm.generate(d["prompts"], params)
    json.dump([[c.text for c in s.outputs] for s in sorties], open(chemin_reponse, "w", encoding="utf-8"), ensure_ascii=False)


if __name__ == "__main__":
    main(*sys.argv[1:3])
