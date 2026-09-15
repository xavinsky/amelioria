"""Édition en masse (MEMIT) de milliers de faits CounterFact avec EasyEdit, sur un GPU 40-80 Go.

    git clone https://github.com/zjunlp/EasyEdit && pip install -e EasyEdit
    python 09-edition-connaissances/infra/memit_easyedit.py --hparams 09-edition-connaissances/infra/memit_llama3-8b.yaml --n 1000

Autres algorithmes EasyEdit avec la même API : ROME, AlphaEdit (édition projetée dans le noyau des
connaissances préservées, meilleure spécificité en édition séquentielle), WISE, GRACE (mémoire externe), MEND.
"""

import argparse
import json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hparams", required=True)
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--sortie", default="sorties/09-edition-connaissances-memit")
    a = p.parse_args()

    from datasets import load_dataset
    from easyeditor import BaseEditor, MEMITHyperParams

    cf = load_dataset("azhx/counterfact", split="train").select(range(a.n))
    rr = [ex["requested_rewrite"] for ex in cf]
    prompts = [r["prompt"].format(r["subject"]) for r in rr]
    editor = BaseEditor.from_hparams(MEMITHyperParams.from_hparams(a.hparams))
    metrics, modele_edite, _ = editor.batch_edit(
        prompts=prompts,
        ground_truth=[r["target_true"]["str"] for r in rr],
        target_new=[r["target_new"]["str"] for r in rr],
        subject=[r["subject"] for r in rr],
        rephrase_prompts=[ex["paraphrase_prompts"][0] for ex in cf],
        locality_inputs={"voisins": {"prompt": [ex["neighborhood_prompts"][0] for ex in cf],
                                     "ground_truth": [r["target_true"]["str"] for r in rr]}},
        sequential_edit=False,
    )
    modele_edite.save_pretrained(a.sortie)
    with open(f"{a.sortie}/metriques.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)


if __name__ == "__main__":
    main()
