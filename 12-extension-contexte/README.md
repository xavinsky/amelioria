# 12 — Extension de la fenêtre de contexte (RoPE : interpolation, NTK, YaRN, ABF)

> Permettre au modèle de traiter des textes plus longs que ceux vus au pré-entraînement : on ajuste l'encodage positionnel, puis un court fine-tuning sur textes longs.

| Regroupe | Modifie | Local 12 Go | Test `--quick` |
|---|---|---|---|
| Position Interpolation (linear), NTK-aware / dynamic NTK, YaRN, LongRoPE, ABF (hausse de `rope_theta`), fine-tuning long contexte | Config RoPE (architecture) + tous les poids | ✅ petits modèles jusqu'à ~8k tokens | ✅ |

## Principe

Les LLM modernes encodent la position par **RoPE** : chaque paire de dimensions de la requête et de la clé tourne d'un angle `position × θ_i`. Au-delà de la longueur d'entraînement, des angles jamais vus apparaissent et la qualité s'effondre.

| `methode` | Idée | Sans fine-tuning ? |
|---|---|---|
| `linear` (Position Interpolation) | Diviser les positions par le facteur : 8k positions deviennent 2k « compressées » | Médiocre, il faut un fine-tuning |
| `dynamic` (NTK dynamique) | Augmenter la base θ seulement quand la séquence dépasse la longueur d'origine | Correct sans entraînement |
| `yarn` | Interpoler uniquement les basses fréquences, garder les hautes, et corriger la température de l'attention | Bon, et excellent après ~400 pas |
| `abf` | Multiplier `rope_theta` (10k → 500k), puis entraîner longtemps sur des textes longs | Non. Recette Llama 3 / Code Llama, pendant le pré-entraînement |
| LongRoPE | Facteurs par dimension trouvés par recherche (Phi-3) | Voir la littérature |

Le réglage n'est **qu'une modification de configuration** (`rope_parameters`). Le fine-tuning sur des séquences longues apprend ensuite au modèle à s'en servir : 0,1 % des tokens du pré-entraînement suffisent pour YaRN.

L'exemple mesure trois états :

1. `origine` : le modèle tel quel ;
2. `echelle_seule` : la config RoPE modifiée, sans entraînement ;
3. `apres_finetuning`.

Pour chacun, on mesure le **test de la clé cachée** (passkey retrieval, retrouver un nombre caché au milieu d'un long texte) à plusieurs longueurs, et la perplexité sur des documents longs.

## Modèles concernés

| Type de modèle | Applicable | Remarques |
|---|---|---|
| LLM avec RoPE (LLaMA, Qwen, Mistral, Gemma, Phi-3, SmolLM, DeepSeek) | ✅ cas principal | [`run.py`](run.py) |
| LLM avec ALiBi (BLOOM, MPT) | ⚠️ | Extrapole un peu nativement ; pas de réglage RoPE, fine-tuning long seulement |
| LLM à positions absolues apprises (GPT-2, OPT) | ⚠️ | Il faut agrandir la table de positions (copie ou interpolation) puis entraîner |
| Encodeurs (BERT 512 → ModernBERT 8k) | ⚠️ | ModernBERT utilise RoPE : même méthode. BERT : positions apprises |
| Mamba / RWKV / hybrides | ✅ | Pas de limite positionnelle : fine-tuning sur textes longs uniquement |
| Vision / vidéo (Qwen2-VL, M-RoPE) | ✅ | RoPE multi-axes, même logique pour plus d'images ou de frames |
| Modèle fermé via API | ❌ | |

## Quand l'utiliser

- ✅ Le modèle doit lire des documents, du code ou des conversations plus longs que sa fenêtre.
- ✅ Adapter un modèle spécialisé (fine-tuné à 4-8k) à du 32k et plus.
- ⚠️ Fenêtre étendue ne veut pas dire usage efficace de tout le contexte : vérifier avec RULER.
- ❌ Si le RAG ([18](../18-rag)) suffit à apporter la bonne information dans une fenêtre courte : c'est bien moins cher à l'inférence.

## Utilisation

```bash
python run.py --quick
python run.py                                        # SmolLM-360M 2k -> 8k, YaRN + 200 pas
python run.py --set contexte.methode=linear
python run.py --set contexte.methode=dynamic --set entrainement.max_steps=0    # NTK dynamique sans entraînement
python run.py --set contexte.methode=abf --set contexte.facteur_theta=20
```

## Paramètres principaux (`config.yaml`)

| Clé | Rôle |
|---|---|
| `contexte.methode`, `facteur` | Type de mise à l'échelle et multiplicateur de longueur |
| `contexte.facteur_theta` | Pour `abf` |
| `entrainement.longueur` | Longueur des séquences d'entraînement (= contexte visé) |
| `evaluation.longueurs` | Longueurs du test de la clé cachée, dont certaines au-delà de la longueur d'origine |

## Grosse infra

- Local : 8k tokens sur un modèle de 360M. La mémoire des activations croît linéairement avec la longueur (flash/SDPA attention) : 32k sur un 7B demande des GPU 80 Go.
- [`infra/config.yaml`](infra/config.yaml) : recette YaRN sur Mistral-7B, 8k → 32k, FSDP sur 8×H100, 400 pas.
- [`infra/axolotl_yarn_128k.yml`](infra/axolotl_yarn_128k.yml) : 128k avec **parallélisme de séquence** (ring-flash-attn, une séquence répartie sur 4 GPU). Indispensable au-delà de ~32k. Alternatives : torchtitan / Megatron « context parallel ».
- Évaluation : RULER, LongBench v2, needle-in-a-haystack, servis par vLLM (`--max-model-len`).

## Pièges

- **Données** : concaténer des paragraphes courts n'apprend pas vraiment les dépendances longues. Utiliser des documents naturellement longs (livres, dépôts de code).
- **Régression en contexte court** : garder des séquences courtes dans le mélange et vérifier les benchmarks habituels.
- **Servir avec la même config** : vLLM et llama.cpp doivent lire le `rope_scaling` sauvegardé. Certaines implémentations appliquent YaRN à toutes les longueurs, ce qui dégrade légèrement le court.
- **Mémoire du cache KV** à l'inférence : elle croît avec la longueur. Voir la quantification du cache ([15](../15-quantification)).

## Références

- Chen et al., *Position Interpolation* (2023) · bloc97, *NTK-aware scaled RoPE* (2023) · Peng et al., *YaRN* (2023)
- Xiong et al., *Effective Long-Context Scaling (ABF)* (2023) · Ding et al., *LongRoPE* (2024)
- Hsieh et al., *RULER* (2024) · Liu et al., *Ring Attention* (2023)
