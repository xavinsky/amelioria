# 07 — Données synthétiques et auto-amélioration

> Faire produire les données d'entraînement par un modèle (lui-même ou un autre), les **filtrer**, puis entraîner dessus.

| Regroupe | Modifie | Local 12 Go | Test `--quick` |
|---|---|---|---|
| Rejection sampling fine-tuning (RFT), STaR / auto-apprentissage, Self-Instruct / Evol-Instruct, critique-révision (phase supervisée de Constitutional AI), best-of-N distillé | Tous les poids (SFT final) | ✅ jusqu'à ~1B | ✅ |

## Principe

Toutes les stratégies suivent le même pipeline en trois temps. Seule l'étape de génération change (`--set strategie=...`) :

```
générer  ->  filtrer / noter  ->  SFT (dossier 02, ou LoRA 03)
```

| `strategie` | Génération | Filtre |
|---|---|---|
| `rejet` (RFT / STaR) | Le modèle lui-même produit k solutions par question | On garde les solutions **vérifiées justes** |
| `self_instruct` | Un modèle générateur invente de **nouvelles questions** à partir d'exemples, puis y répond | Réponse vérifiable, déduplication, format valide |
| `critique_revision` | Le modèle répond, se critique selon un **principe écrit**, puis réécrit sa réponse | On garde la révision (entraînement sur « réponse révisée ») |

La clé est **le filtre**. Un modèle qui s'entraîne sur ses propres sorties non filtrées stagne ou se dégrade (model collapse). Filtrer par un vérificateur, un juge ou un modèle de récompense ne garde que ses meilleures sorties : le modèle s'améliore ainsi « par lui-même ». C'est le même ressort que le RL en ligne ([05](../05-rl-en-ligne)), en version hors ligne, plus simple et plus stable.

Lien avec la distillation ([06](../06-distillation)) : si le générateur est un modèle plus fort que l'élève, `self_instruct` et `rejet` deviennent de la distillation de séquences.

## Modèles concernés

| Type de modèle | Applicable | Remarques |
|---|---|---|
| LLM décodeur instruct | ✅ cas principal | Le modèle doit réussir la tâche au moins de temps en temps (`rejet`), ou savoir suivre des consignes (`critique_revision`) |
| LLM de base | ⚠️ | Utiliser un générateur instruct distinct (`generateur.nom`) |
| Encodeur / classifieur | ✅ | Un LLM génère des textes étiquetés pour entraîner un petit classifieur (variante courante de `self_instruct`, voir *Grosse infra*) |
| Vision / multimodal | ✅ | Légendes générées par un VLM et filtrées par CLIP-score (recette de nombreux jeux image-texte) |
| Diffusion | ✅ | Prompts réécrits par un LLM (légendes synthétiques détaillées, recette DALL-E 3) |
| Modèle fermé via API | ✅ comme générateur | Générer avec GPT/Claude/Gemini pour entraîner un modèle ouvert, en respectant leurs conditions d'utilisation |

## Quand l'utiliser

- ✅ Peu ou pas de données annotées, mais une **tâche vérifiable** ou un bon générateur disponible.
- ✅ Spécialiser un petit modèle bon marché à partir d'un gros (souvent la méthode la plus rentable en production).
- ✅ Aligner sur des principes écrits sans annotateurs (`critique_revision`).
- ❌ Sans filtre fiable : le risque de dégradation l'emporte.

## Utilisation

```bash
python run.py --quick
python run.py                                                  # rejection sampling, Qwen2.5-0.5B s'auto-entraîne
python run.py --set strategie=self_instruct --set generateur.nom=Qwen/Qwen2.5-1.5B-Instruct
python run.py --set strategie=critique_revision
python run.py --set tours=3                                    # itérer (STaR / ReST-EM)
```

Le jeu généré est sauvegardé dans `sorties/.../donnees-tour-N.jsonl`, réutilisable par [02](../02-fine-tuning-supervise) ou [03](../03-peft) (`donnees.source=hf`, `donnees.nom=json`, `donnees.fichiers=...`).

## Ce que fait l'exemple

- **`rejet`** (par défaut) :
  1. Pour 1 000 questions, le modèle produit 4 solutions échantillonnées.
  2. On garde une solution juste par question résolue.
  3. SFT du modèle sur ces solutions.
  4. Remesure.

  Le rapport donne le taux de questions résolues à chaque tour : il monte d'un tour à l'autre.
- **`self_instruct`** : le générateur écrit de nouveaux problèmes du même style. On ne garde que ceux dont l'expression se calcule, et la réponse vient du calcul lui-même, donc elle est garantie juste.
- **`critique_revision`** : un principe (« être concis et donner le résultat final sous la forme Answer: N ») guide la critique puis la réécriture.

## Paramètres principaux (`config.yaml`)

| Clé | Rôle |
|---|---|
| `strategie` | `rejet`, `self_instruct` ou `critique_revision` |
| `generateur.nom` | `null` = le modèle entraîné lui-même (auto-amélioration) ; sinon un modèle plus fort |
| `generation.k`, `temperature` | Diversité des candidats |
| `tours` | Nombre d'itérations générer / filtrer / entraîner |
| `principes` | Liste de principes pour `critique_revision` |

## Grosse infra

- La génération domine le coût : **vLLM** sur un nœud (script [`../06-distillation/infra/generer_vllm.py`](../06-distillation/infra/generer_vllm.py), rejection sampling inclus), puis SFT FSDP ([02 infra](../02-fine-tuning-supervise/infra)).
- [`infra/lancer.sh`](infra/lancer.sh) : boucle STaR / ReST-EM à grande échelle (génération vLLM, filtre, SFT, et on recommence), plus les outils de génération de données dédiés : [distilabel](https://github.com/argilla-io/distilabel), [DataDreamer](https://github.com/datadreamer-dev/DataDreamer), NVIDIA NeMo Curator / Nemotron-4-340B pour la génération et la notation.
- Ordres de grandeur : les jeux synthétiques récents (OpenMathInstruct-2, 14 M solutions ; Tülu 3 SFT mix) demandent des milliers d'heures GPU de génération.

## Pièges

- **Model collapse** : jamais sans filtre. Garder une part de données réelles.
- **Faux positifs du vérificateur** : bonne réponse obtenue par un raisonnement faux. Préférer plusieurs solutions justes et diverses.
- **Diversité** : Self-Instruct converge vite vers les mêmes questions. Dédupliquer (n-grammes, embeddings), varier les exemples d'amorce.
- **Contamination** : vérifier que les données générées ne recopient pas le jeu de test.
- **Licences** : les conditions de certains fournisseurs interdisent d'entraîner un modèle concurrent sur leurs sorties.

## Références

- Zelikman et al., *STaR* (2022) · Yuan et al., *Scaling Relationship… RFT* (2023) · Singh et al., *ReST-EM* (2023)
- Wang et al., *Self-Instruct* (2022) · Xu et al., *WizardLM / Evol-Instruct* (2023)
- Bai et al., *Constitutional AI* (2022) : phase critique-révision · Shumailov et al., *Model collapse* (2024)
