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

Depuis la racine : `make test EX=NN [VAR=...] [SET="cle=val ..."]` (test rapide) et `make run EX=NN` (complet) vérifient les prérequis puis lancent `run.py` ; voir `make help`. Directement :

```bash
python NN-nom/run.py                         # config.yaml (GPU local ~12 Go)
python NN-nom/run.py --quick                 # petit modèle, peu de données, quelques pas : vérifie que tout s'exécute
python NN-nom/run.py --set cle.sous_cle=val  # surcharge (valeur lue en YAML : nombres, listes, null... ; « 5e-6 » est un nombre, « 2:4 » du texte)
python NN-nom/run.py --config NN-nom/infra/config.yaml
python NN-nom/run.py --sortie /chemin        # dossier de sortie
bash   NN-nom/infra/lancer.sh                # grosse infra
```

Choix d'une variante d'algorithme : toujours par la config (`--set methode=...`, `--set algo=...`, `--set peft.methode=...`).

## Mêmes clés de configuration

| Section | Contenu |
|---|---|
| `modele` | `nom` (id Hugging Face ou dossier local), `dtype`, `api_base` pour les méthodes EXEC17 à EXEC20 |
| `donnees` | `source` (`banking77`, `arith`, `faits`, `gsm8k`, `hf`...), tailles, `graine`, `remelange` (part de demandes générales mêlées à l'entraînement contre l'oubli) |
| `entrainement` | Clés courtes traduites en `TrainingArguments` : `lr`, `epochs`, `max_steps`, `batch`, `accumulation`, `warmup`, `scheduler`, `gradient_checkpointing` (désactivé si absent, alors que les configs TRL l'activent par défaut), `optim` (ex. `adamw_bnb_8bit`), `logging_steps`, `autres: {...}` pour tout le reste |
| `evaluation` | `mode` (`etiquette`, `nombre` ou `contient`), `n`, `max_new_tokens`, `systeme` (`{intentions}` y est remplacé par la liste des réponses possibles), `non_regression` (tâche hors sujet mesurée avant et après), `perplexite_generale`, `activer` |
| `sauvegarder` | Écrire ou non le modèle produit |
| `quick` | Surcharges du test rapide |

## Mêmes sorties

`sorties/<dossier>[-quick]/` contient :

- `config_effective.yaml` : la configuration réellement utilisée ;
- `resultats.json` : métriques de chaque étape (`avant`, `apres`, ou étapes nommées), et `comparaisons` quand des étapes ont répondu aux mêmes questions ;
- le modèle ou l'adaptateur produit.

Le terminal affiche un tableau « métrique × étape » identique d'un exemple à l'autre. Chaque exactitude y porte son intervalle de confiance à 95 % (Wilson) : `0.67 [0.602, 0.732]`.

## Mêmes outils de preuve

- **Comparaison appariée.** Quand deux étapes ont répondu aux mêmes questions (avant et après, avec et sans RAG, deux stratégies de prompt), chacune est comparée à la première étape, question par question : différence d'exactitude, intervalle à 95 % par bootstrap apparié, p-valeur du test exact de McNemar. Le rapport conclut « écart net » quand l'intervalle exclut 0.
- **Graines.** `make graines EX=… [N=3] [QUICK=1]` répète l'exemple avec les graines 42, 43… (la première reproduit l'exécution par défaut), puis affiche moyenne, écart type, minimum et maximum de chaque métrique, comparaisons comprises. Agrégat : `sorties/<dossier>--graines/resultats.json`. La graine change l'initialisation, l'ordre des données d'entraînement et les tirages ; les jeux de test restent les mêmes.

Un gain est démontré quand l'intervalle de la différence exclut 0 et que le résultat se reproduit sur plusieurs graines pour les entraînements courts (critères : [cahier-des-charges.md](cahier-des-charges.md)).

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
| `config.py` | Ligne de commande, fusion `quick`, `--set`, dossier de sortie, graine (reprise par les arguments des Trainer) |
| `modeles.py` | Chargement de LLM (bf16/fp32, bitsandbytes, multi-GPU), prompts, génération, log-probs, couches. `sans_autocast` retire l'autocast qu'accelerate laisse sur le modèle après un entraînement : les mesures avant/après se font dans la même précision |
| `donnees.py` | Jeux fil rouge, banking77, demandes générales du remélange, chargeurs Hugging Face |
| `evaluation.py` | Exactitude (avec intervalle à 95 % et détail par question), perplexité, évaluation pilotée par la config |
| `preuve.py` | Intervalle de Wilson, comparaison appariée (bootstrap, McNemar), écart moyen apparié d'un score continu (bootstrap, permutation), agrégation de plusieurs graines |
| `entrainement.py` | Arguments communs, SFT (TRL) avec durée et pic de mémoire GPU, remélange, LM causal, tokenisation pour boucles manuelles, LoRA. `liberer` rend la mémoire de l'optimiseur après l'entraînement |
| `generateur.py` | Chat unifié modèle local / API compatible OpenAI |
| `rapport.py` | `resultats.json` + tableau, intervalles et comparaisons appariées |
| `infra/` | accelerate FSDP2 et DeepSpeed ZeRO-3, Slurm multi-nœuds, évaluation lm-eval + vLLM |
