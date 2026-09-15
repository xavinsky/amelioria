# 14 — Croissance d'architecture (depth up-scaling, block expansion, MoE upcycling)

> Agrandir un modèle déjà entraîné (plus de couches, ou des experts) puis le « cicatriser » par un entraînement continu. Beaucoup moins cher que de pré-entraîner le grand modèle de zéro.

| Regroupe | Modifie | Local 12 Go | Test `--quick` |
|---|---|---|---|
| Depth up-scaling (SOLAR), block expansion (LLaMA Pro), sparse upcycling dense → MoE (Mixtral), MoE de fine-tunes (mergekit-moe) | Architecture (nombre de couches, experts) + poids | ✅ transformation de n'importe quelle taille (CPU) ; cicatrisation locale jusqu'à ~1B | ✅ |

## Principe

On réutilise les poids existants comme point de départ d'une architecture plus grande. Les trois méthodes suivent le même déroulé :

```
transformer l'architecture  ->  mesurer l'effet immédiat  ->  pré-entraînement continu  ->  remesurer
```

| `methode` | Transformation | État juste après | Ce qu'on entraîne |
|---|---|---|---|
| `profondeur` (SOLAR) | n couches → `[0, n-m)` + `[m, n)` : 2(n-m) couches, les couches centrales dupliquées se chevauchent | Dégradé (couture au milieu), récupération rapide | Tout |
| `expansion_blocs` (LLaMA Pro) | Après chaque groupe de k couches, une copie dont `o_proj` et `down_proj` sont **à zéro** : la couche est une identité | **Strictement identique** à l'original (`ecart_max_logits_vs_origine` ≈ 0) | Seulement les nouvelles couches : les connaissances d'origine sont préservées |
| `moe` (sparse upcycling) | Chaque MLP dense devient E experts identiques + un routeur top-k (LLaMA → Mixtral) | Identique à l'original (experts égaux, poids du routeur normalisés) | Tout. Les experts se différencient pendant l'entraînement |

L'intérêt : plus de capacité (profondeur) ou plus de paramètres pour un coût d'inférence presque constant (MoE top-2 sur E experts), en partant d'un modèle qui sait déjà beaucoup de choses.

Le rapport contient :

- les paramètres et le nombre de couches ;
- la perplexité générale à chaque étape ;
- l'écart maximal de logits avec l'original (sanity check des transformations censées préserver la fonction).

## Modèles concernés

| Type de modèle | Applicable | Remarques |
|---|---|---|
| LLM famille LLaMA / Mistral / SmolLM (sans biais) | ✅ cas principal | [`croissance.py`](croissance.py) : les trois méthodes |
| Qwen2, Gemma, Phi | ⚠️ | `profondeur` et `expansion_blocs` fonctionnent si les noms de modules sont les mêmes. `moe` : la cible est l'équivalent MoE de la famille (Qwen2 → Qwen2-MoE, avec biais d'attention) |
| Encodeurs (BERT) / ViT | ✅ | Principe identique : empilement (stacking de BERT), expansion identité |
| Vision / multimodal | ✅ | Upcycling MoE de ViT (V-MoE) et de VLM (MoE-LLaVA) |
| Diffusion | ⚠️ | Rare, mais des DiT MoE existent |
| Modèle fermé via API | ❌ | |

## Quand l'utiliser

- ✅ Obtenir un modèle plus capable qu'un checkpoint existant sans budget de pré-entraînement complet.
- ✅ `expansion_blocs` : ajouter un domaine (code, médecine) en préservant les capacités d'origine.
- ✅ `moe` : augmenter la capacité à coût d'inférence quasi constant.
- ⚠️ La cicatrisation reste un **pré-entraînement continu conséquent** : milliards de tokens pour des modèles de 7B et plus.
- ❌ Sans budget d'entraînement : le modèle transformé n'est pas meilleur que l'original (`profondeur` est même pire).

## Utilisation

```bash
python run.py --quick
python run.py                                                   # SmolLM2-360M 32 -> 48 couches (SOLAR) + 600 pas
python run.py --set croissance.methode=expansion_blocs          # 32 -> 40 couches, seules les 8 nouvelles apprennent
python run.py --set croissance.methode=moe --set croissance.n_experts=4
python run.py --set croissance.methode=profondeur --set croissance.m=12
```

## Paramètres principaux (`config.yaml`)

| Clé | Rôle |
|---|---|
| `croissance.methode` | `profondeur`, `expansion_blocs` ou `moe` |
| `croissance.m` | Chevauchement SOLAR : couches totales = 2(n-m) |
| `croissance.tous_les` | Fréquence des blocs identité |
| `croissance.n_experts`, `top_k`, `bruit_experts` | Configuration MoE |
| `entrainement.*` | Cicatrisation (pré-entraînement continu) |

## Grosse infra

- La **transformation** est instantanée et se fait sur CPU, même pour un 70B :
  - [`infra/mergekit_passthrough_solar.yml`](infra/mergekit_passthrough_solar.yml) : Mistral-7B → 10,7B, sans code ;
  - [`infra/mergekit_moe.yml`](infra/mergekit_moe.yml) : MoE 4×7B à partir de 4 fine-tunes *différents*, routeur initialisé par des prompts.
- La **cicatrisation** est un pré-entraînement continu multi-nœuds ([`infra/config.yaml`](infra/config.yaml), [`infra/lancer.sh`](infra/lancer.sh), ou [01 infra](../01-pre-entrainement-continu/infra)).
- Pour l'entraînement de MoE à grande échelle, le **parallélisme d'experts** devient nécessaire : Megatron-LM / NeMo (upcycling intégré), torchtitan, DeepSpeed-MoE.

## Pièges

- **Indices de couche** : après duplication, `layer_idx` doit être renuméroté, sinon le cache KV est corrompu à la génération. C'est fait dans `_renumeroter`.
- **SOLAR sans cicatrisation** : perplexité nettement dégradée. C'est normal.
- **MoE** : sans perte auxiliaire d'équilibrage (`router_aux_loss_coef`), le routeur envoie tout vers 1 ou 2 experts. Des experts identiques mettent du temps à se différencier (bruit, données variées).
- **Mémoire** : un MoE 4 experts pèse ~3 fois le dense (les MLP dominent). L'inférence active seulement top-k experts, mais tous doivent être chargés.

## Références

- Kim et al., *SOLAR 10.7B: Depth Up-Scaling* (2023) · Wu et al., *LLaMA Pro: Progressive LLaMA with Block Expansion* (2024)
- Komatsuzaki et al., *Sparse Upcycling* (2022) · Jiang et al., *Mixtral of Experts* (2024) · He et al., *Upcycling LLMs into MoE* (NVIDIA, 2024)
- Gong et al., *Efficient Training of BERT by Progressively Stacking* (2019)
