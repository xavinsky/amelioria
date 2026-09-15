"""Extension « propre » d'un tokenizer BPE pour une langue : apprendre un BPE sur le corpus cible puis
fusionner ses tokens et ses règles de fusion dans le tokenizer existant (au lieu de simples added_tokens).

    python 13-extension-vocabulaire/infra/entrainer_tokenizer.py --modele meta-llama/Llama-3.1-8B \
        --dataset wikimedia/wikipedia --config 20231101.ko --taille 16000 --sortie sorties/tokenizer-ko
"""

import argparse
import json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--modele", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--config", default=None)
    p.add_argument("--colonne", default="text")
    p.add_argument("--n", type=int, default=200000)
    p.add_argument("--taille", type=int, default=16000)
    p.add_argument("--sortie", required=True)
    a = p.parse_args()

    from datasets import load_dataset
    from transformers import AutoTokenizer

    base = AutoTokenizer.from_pretrained(a.modele)
    ds = load_dataset(a.dataset, a.config, split="train", streaming=True)
    textes = (ex[a.colonne] for _, ex in zip(range(a.n), ds))
    cible = base.train_new_from_iterator(textes, vocab_size=a.taille)   # même pré-tokenisation et normalisation

    etat_base = json.loads(base.backend_tokenizer.to_str())
    etat_cible = json.loads(cible.backend_tokenizer.to_str())
    vocab, merges = etat_base["model"]["vocab"], etat_base["model"]["merges"]
    connus = set(tuple(m) if isinstance(m, list) else tuple(m.split(" ")) for m in merges)
    ajoutes = 0
    for m in etat_cible["model"]["merges"]:
        paire = tuple(m) if isinstance(m, list) else tuple(m.split(" "))
        jeton = "".join(paire)
        if paire in connus or jeton in vocab or any(x not in vocab for x in paire):
            continue
        vocab[jeton] = len(vocab)
        merges.append(list(paire) if isinstance(m, list) else m)
        connus.add(paire)
        ajoutes += 1
    from tokenizers import Tokenizer

    base._tokenizer = Tokenizer.from_str(json.dumps(etat_base))
    base.save_pretrained(a.sortie)
    print(f"{ajoutes} tokens ajoutés -> {a.sortie} ; attention : les ids des tokens spéciaux ajoutés après le vocabulaire BPE sont à vérifier")


if __name__ == "__main__":
    main()
