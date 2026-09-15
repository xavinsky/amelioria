# 20 — Calcul à l'inférence (self-consistency, best-of-N, auto-correction)

> Dépenser plus de calcul **par question** au moment de répondre (plusieurs tentatives, sélection, vérification) plutôt que davantage d'entraînement.

| Regroupe | Modifie | Local 12 Go | Test `--quick` |
|---|---|---|---|
| Self-consistency (vote majoritaire), best-of-N par modèle de récompense, best-of-N par vraisemblance, self-refine (auto-correction), borne oracle pass@N. En infra : recherche guidée par PRM, budget de réflexion des modèles de raisonnement | Rien : la procédure de décodage | ✅ | ✅ |

## Principe

Un modèle qui se trompe une fois sur deux en génération gloutonne produit souvent la bonne réponse **dans au moins une** de ses N tentatives. Tout l'enjeu est de la **retrouver** :

| Stratégie | Comment on choisit parmi N réponses | Besoin |
|---|---|---|
| `glouton` | Pas de choix : une seule réponse à température 0 | Référence |
| `vote_majoritaire` | La réponse finale la plus fréquente (self-consistency) | Une réponse finale comparable (nombre, choix, code exécutable) |
| `best_of_n_vraisemblance` | Log-probabilité moyenne la plus haute | Accès aux log-probs |
| `best_of_n_recompense` | Score d'un **modèle de récompense** ou d'un vérificateur | Un RM ([05 variante](../05-rl-en-ligne/variantes/modele-recompense)) ou un vérificateur exécutable |
| `auto_correction` | Le modèle relit sa réponse et la corrige (self-refine) | Un modèle capable de se critiquer |
| `oracle_pass_a_N` | *Une* réponse juste existe-t-elle parmi les N ? | Borne supérieure de toute stratégie de sélection |

Les stratégies de sélection partagent **les mêmes N échantillons**. La courbe `vote_majoritaire.courbe` (N = 1, 2, 4, 8, 16) montre la loi d'échelle : le gain rapide des premiers échantillons, puis le plateau.

Les **modèles de raisonnement** (o1/o3, DeepSeek-R1, Qwen3 en mode thinking) intègrent cette idée dans les poids : entraînés par RL ([05](../05-rl-en-ligne)), ils génèrent eux-mêmes une longue réflexion avec vérifications et retours en arrière. Le « budget de réflexion » (nombre maximal de tokens) devient un réglage de calcul à l'inférence.

## Modèles concernés

| Type de modèle | Applicable | Remarques |
|---|---|---|
| LLM instruct, local ou via API | ✅ cas principal | `best_of_n_vraisemblance` demande un modèle local (log-probs) |
| Modèles de raisonnement | ✅ | Budget de réflexion. Le vote reste utile au-dessus |
| Code | ✅ très efficace | Sélection par exécution des tests (vérificateur exact) |
| Multimodal | ✅ | Mêmes stratégies |
| Diffusion | ✅ | Best-of-N d'images notées par un modèle esthétique ou d'alignement (ImageReward, PickScore) ; recherche sur le bruit initial |
| Encodeurs / classifieurs | ⚠️ | Équivalent : test-time augmentation et ensembles |

## Quand l'utiliser

- ✅ Tâches à réponse vérifiable ou comparable (maths, code, QCM, extraction).
- ✅ Quand la latence et le coût par requête sont acceptables : N générations coûtent environ N fois plus.
- ✅ Pour mesurer le **potentiel** d'un modèle avant de l'entraîner : un écart important entre `glouton` et `oracle_pass_a_N` indique un modèle qui gagnera beaucoup avec du RL ou du rejection sampling ([05](../05-rl-en-ligne), [07](../07-donnees-synthetiques)).
- ❌ Réponses ouvertes non comparables (rédaction) : le vote ne s'applique pas, il faut un juge ou un RM.

## Utilisation

```bash
python run.py --quick
python run.py                                                   # GSM8K, Qwen2.5-1.5B-Instruct, N=16
python run.py --set n_echantillons=32 --set temperature=1.0
python run.py --set modele_recompense=sorties/05-rl-en-ligne--variantes--modele-recompense/modele --set donnees.source=arith
python run.py --set modele.api_base=http://localhost:8000/v1 --set modele.nom=Qwen/Qwen2.5-7B-Instruct --set "strategies=[vote_majoritaire, auto_correction]"
```

## Paramètres principaux (`config.yaml`)

| Clé | Rôle |
|---|---|
| `n_echantillons`, `temperature` | Budget et diversité |
| `strategies` | Stratégies évaluées |
| `modele_recompense` | Pour `best_of_n_recompense` |
| `tours_correction` | Passes d'auto-correction |

## Grosse infra

- La génération domine : **vLLM** en parallélisme de données (`--data-parallel-size 8`, un réplica par GPU) avec cache de préfixe. Les N échantillons d'une même question partagent le prompt.
- [`infra/config.yaml`](infra/config.yaml) : GSM8K complet (1 319 questions), N=64, RM Skywork 8B ; [`infra/lancer.sh`](infra/lancer.sh).
- Recherche guidée par **Process Reward Model** (note chaque étape : beam search, DVTS) : [search-and-learn](https://github.com/huggingface/search-and-learn).

## Pièges

- **Coût** : N=16, c'est 16 fois plus de tokens. Comparer les stratégies à budget égal.
- **Vote sur réponses non normalisées** : « 72 », « 72.0 » et « $72 » doivent compter pareil (voir `extraire_nombre`).
- **Reward hacking à l'inférence** : avec N grand, le best-of-N par RM finit par choisir des réponses qui trompent le RM. Le vote est plus robuste.
- **Auto-correction des petits modèles** : ils « corrigent » souvent une bonne réponse en mauvaise. Mesurer avant d'adopter.

## Références

- Wang et al., *Self-Consistency* (2022) · Cobbe et al., *Training Verifiers (GSM8K)* (2021) · Madaan et al., *Self-Refine* (2023)
- Lightman et al., *Let's Verify Step by Step (PRM)* (2023) · Snell et al., *Scaling LLM Test-Time Compute Optimally* (2024)
- Brown et al., *Large Language Monkeys: Scaling Inference Compute with Repeated Sampling* (2024) · DeepSeek-AI, *DeepSeek-R1* (2025)
