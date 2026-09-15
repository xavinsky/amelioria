# Structure commune des exemples

## Un dossier par méthode (ou groupe de méthodes)

```
NN-nom/
  README.md        mêmes sections partout (voir plus bas)
  run.py           exemple principal : mesurer -> appliquer la méthode -> remesurer -> sauvegarder
  config.yaml      réglages « poste local » + section quick: (test rapide)
  *.py             algorithmes propres à la méthode quand ils méritent d'être lus à part (fusion.py, edition.py...)
  variantes/<nom>/ autres types de modèles ou autres recettes : même paire run.py + config.yaml
  infra/           config.yaml + lancer.sh pour grosse infra, et fichiers des outils dédiés (mergekit, verl, EasyEdit...)
```

## Même ligne de commande partout

```bash
python NN-nom/run.py                         # config.yaml (GPU local ~12 Go)
python NN-nom/run.py --quick                 # petit modèle, peu de données, quelques pas : vérifie que tout s'exécute
python NN-nom/run.py --set cle.sous_cle=val  # surcharge (valeur lue en YAML : nombres, listes, null...)
python NN-nom/run.py --config NN-nom/infra/config.yaml
python NN-nom/run.py --sortie /chemin        # dossier de sortie
bash   NN-nom/infra/lancer.sh                # grosse infra
```

Choix d'une variante d'algorithme : toujours par la config (`--set methode=...`, `--set algo=...`, `--set peft.methode=...`).

## Mêmes clés de configuration

| Section | Contenu |
|---|---|
| `modele` | `nom` (id Hugging Face ou dossier local), `dtype`, `api_base` pour les méthodes 17 à 20 |
| `donnees` | `source` (`arith`, `faits`, `gsm8k`, `hf`...), tailles, `graine` |
| `entrainement` | Clés courtes traduites en `TrainingArguments` : `lr`, `epochs`, `max_steps`, `batch`, `accumulation`, `warmup`, `scheduler`, `gradient_checkpointing`, `logging_steps`, `autres: {...}` pour tout le reste |
| `evaluation` | `mode` (`nombre` ou `contient`), `n`, `max_new_tokens`, `perplexite_generale`, `activer` |
| `sauvegarder` | Écrire ou non le modèle produit |
| `quick` | Surcharges du test rapide |

## Mêmes sorties

`sorties/<dossier>[-quick]/` contient :

- `config_effective.yaml` : la configuration réellement utilisée ;
- `resultats.json` : métriques de chaque étape (`avant`, `apres`, ou étapes nommées) ;
- le modèle ou l'adaptateur produit.

Le terminal affiche un tableau « métrique × étape » identique d'un exemple à l'autre.

## Mêmes données fil rouge

| Source | Contenu | Utilisée par |
|---|---|---|
| `arith` | Problèmes `(a op b) op c` avec raisonnement et `Answer: N` ; réponse vérifiable | 02, 03, 04, 05, 06, 07, 15, 16, 17, 20 |
| `faits` | Personnages fictifs (ville, année, métier, création) en 4 formulations + questions | 01, 10, 13, 18, 19 |
| `EDITIONS` | Faits réels à modifier, avec paraphrases et voisins | 09 |
| wikitext | Texte général : perplexité, replay, calibration, cicatrisation | 01, 12, 14, 15, 16 |

Les jeux réels (GSM8K, UltraFeedback, TOFU, SST-2, CIFAR-10, MLS...) se branchent par la config.

## Socle `commun/`

| Module | Rôle |
|---|---|
| `config.py` | Ligne de commande, fusion `quick`, `--set`, dossier de sortie, graine |
| `modeles.py` | Chargement de LLM (bf16/fp32, bitsandbytes, multi-GPU), prompts, génération, log-probs, couches |
| `donnees.py` | Jeux fil rouge et chargeurs Hugging Face |
| `evaluation.py` | Exactitude, perplexité, évaluation pilotée par la config |
| `entrainement.py` | Arguments communs, SFT (TRL), LM causal, tokenisation pour boucles manuelles, LoRA |
| `generateur.py` | Chat unifié modèle local / API compatible OpenAI |
| `rapport.py` | `resultats.json` + tableau |
| `infra/` | accelerate FSDP2 et DeepSpeed ZeRO-3, Slurm multi-nœuds, évaluation lm-eval + vLLM |
