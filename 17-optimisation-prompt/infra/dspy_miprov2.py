"""Optimisation de programme LLM avec DSPy (MIPROv2) : instructions ET exemples few-shot optimisés conjointement,
par optimisation bayésienne sur un jeu de validation. Fonctionne avec n'importe quel modèle servi (vLLM, API).

    pip install dspy
    python 17-optimisation-prompt/infra/dspy_miprov2.py --modele openai/Qwen/Qwen2.5-72B-Instruct --api-base http://localhost:8000/v1
"""

import argparse
import pathlib
import sys

sys.path.insert(0, str(next(p for p in pathlib.Path(__file__).resolve().parents if (p / "commun").is_dir())))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--modele", required=True, help="identifiant LiteLLM, ex. openai/<nom servi>")
    p.add_argument("--api-base", default=None)
    p.add_argument("--n-train", type=int, default=300)
    p.add_argument("--n-test", type=int, default=500)
    a = p.parse_args()

    import dspy

    from commun import donnees

    dspy.configure(lm=dspy.LM(a.modele, api_base=a.api_base, api_key="x", max_tokens=512))
    exemples = lambda qa: [dspy.Example(question=x["question"], answer=x["cible"]).with_inputs("question") for x in qa]
    train, test = exemples(donnees.gsm8k(a.n_train, "train")), exemples(donnees.gsm8k(a.n_test, "test"))

    programme = dspy.ChainOfThought("question -> answer: str")
    metrique = lambda ex, pred, trace=None: donnees.extraire_nombre(str(pred.answer)) == ex.answer

    evaluer = dspy.Evaluate(devset=test, metric=metrique, num_threads=32, display_progress=True)
    print("avant :", evaluer(programme))
    optimise = dspy.MIPROv2(metric=metrique, auto="medium", num_threads=32).compile(programme, trainset=train)
    print("après :", evaluer(optimise))
    optimise.save("sorties/17-optimisation-prompt-dspy.json")


if __name__ == "__main__":
    main()
