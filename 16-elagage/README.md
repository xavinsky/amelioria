# 16 — Élagage (pruning) et cicatrisation

> Retirer les poids, neurones ou couches qui contribuent le moins, puis récupérer la qualité par un court entraînement, idéalement une distillation depuis le modèle d'origine.

| Regroupe | Modifie | Local 12 Go | Test `--quick` |
|---|---|---|---|
| Non structuré : magnitude, Wanda (SparseGPT en infra) ; semi-structuré 2:4 ; structuré : suppression de couches (ShortGPT), réduction de largeur MLP (Minitron) ; cicatrisation LM ou distillation | Poids mis à zéro, ou architecture réduite | ✅ jusqu'à ~1,5B avec cicatrisation, ~7B sans | ✅ |

## Principe

| `methode` | Ce qui est retiré | Gain réel |
|---|---|---|
| `magnitude` | Les poids de plus petite valeur absolue, dans chaque ligne | Aucun gain de vitesse sans moteur sparse. Référence de comparaison |
| `wanda` | Score `|W_ij| · ‖X_j‖₂` : un petit poids sur une entrée à forte activation reste important. Calibration sans rétropropagation | Idem, mais bien meilleure qualité que magnitude à 50 % |
| `2:4` | Dans chaque groupe de 4 poids, les 2 moins importants (score Wanda) | Accélération réelle ~1,3 à 1,6× sur GPU Ampere et plus récents (noyaux 2:4, vLLM) |
| `couches` | Les blocs dont la sortie ressemble le plus à l'entrée (Block Influence = 1 - cos) | Modèle plus petit et plus rapide, proportionnellement |
| `largeur` | Les neurones MLP les moins activés : `gate`/`up` perdent des lignes, `down` des colonnes | Modèle plus petit et plus rapide |

La **cicatrisation** (`cicatrisation.mode`) est ce qui rend l'élagage agressif utilisable :

- `lm` : pré-entraînement continu sur du texte général ;
- `distillation` : le modèle élagué imite la distribution de sortie de l'original (recette Minitron de NVIDIA). Plus efficace que `lm` à nombre de tokens égal.

Pendant la cicatrisation d'un élagage non structuré, un masque (`torch.nn.utils.prune`) garantit que les zéros restent des zéros. Il est « figé » dans les poids à la fin.

Chaque étape est mesurée (origine → élagué → cicatrisé) : paramètres, sparsité, tokens/s, perplexité générale, exactitude sur la tâche.

## Modèles concernés

| Type de modèle | Applicable | Remarques |
|---|---|---|
| LLM LLaMA / Qwen / Mistral / SmolLM | ✅ cas principal | [`elagage.py`](elagage.py) : noms `q/k/v/o_proj`, `gate/up/down_proj` |
| Autres LLM (GPT-2, Phi, Falcon) | ⚠️ | Adapter `NOMS_LINEAIRES` et les attributs MLP |
| MoE | ✅ | Élaguer des experts entiers (fréquence de routage) : très efficace |
| Encodeurs (BERT) | ✅ très étudié | Élagage de têtes d'attention, movement pruning, `nn_pruning` |
| Vision (CNN, ViT) | ✅ | Élagage de filtres / canaux (torch-pruning), tokens (ToMe) |
| Diffusion | ✅ | Suppression de blocs de l'UNet + distillation (BK-SDM, SSD-1B) |
| Modèle fermé via API | ❌ | |

## Quand l'utiliser

- ✅ Produire une famille de modèles plus petits à partir d'un grand, avec distillation. Moins cher qu'un pré-entraînement : Llama-3.1-Minitron 4B.
- ✅ Accélérer l'inférence avec 2:4 ou l'élagage structuré.
- ⚠️ Non structuré sans moteur sparse : aucun gain de mémoire ni de vitesse en PyTorch standard. Intérêt surtout pour la compression combinée à la quantification.
- ❌ Élagage structuré agressif sans budget de cicatrisation : la qualité s'effondre.

## Utilisation

```bash
python run.py --quick
python run.py                                                  # Wanda 50 % + distillation, Qwen2.5-1.5B-Instruct
python run.py --set methode=magnitude --set cicatrisation.mode=aucune    # comparer à Wanda sans cicatrisation
python run.py --set methode=2:4
python run.py --set methode=couches --set n_couches=6
python run.py --set methode=largeur --set ratio_largeur=0.3 --set cicatrisation.mode=lm
```

## Paramètres principaux (`config.yaml`)

| Clé | Rôle |
|---|---|
| `methode` | Voir le tableau |
| `sparsite`, `n_couches`, `ratio_largeur` | Intensité selon la méthode |
| `calibration.n` | Textes de calibration (Wanda, importance des couches et des neurones) |
| `cicatrisation.mode`, `entrainement.*` | Récupération |

## Grosse infra

- L'**élagage** reste léger : Wanda ou SparseGPT sur un 70B se font couche par couche sur 1 à 2 GPU (llm-compressor, [`infra/lancer.sh`](infra/lancer.sh)).
- La **cicatrisation par distillation** est coûteuse. Minitron distille sur ~100 Md de tokens pour un 8B → 4B : c'est un travail multi-nœuds ([`infra/config.yaml`](infra/config.yaml), Slurm).
- Outillage industriel : NVIDIA TensorRT Model Optimizer + NeMo (élagage profondeur, largeur, têtes, dimension cachée ; distillation), llm-compressor (SparseGPT 2:4 → vLLM).

## Pièges

- **Mesurer le vrai gain** : 50 % de zéros non structurés ne rendent ni plus petit ni plus rapide sans format sparse.
- **Couches retirées** : les premières et la dernière couche sont presque toujours critiques. Le score d'influence les protège en général.
- **Élagage après quantification** (ou l'inverse) : l'ordre compte. L'usage courant est d'élaguer, cicatriser, puis quantifier.
- **Cache KV et `layer_idx`** : après suppression de couches, les indices doivent être renumérotés. C'est fait dans `supprimer_couches`.

## Références

- Han et al., *Learning both Weights and Connections* (2015) · Frantar & Alistarh, *SparseGPT* (2023) · Sun et al., *Wanda* (2023)
- Men et al., *ShortGPT* (2024) · Gromov et al., *The Unreasonable Ineffectiveness of the Deeper Layers* (2024)
- Muralidharan et al., *Compact Language Models via Pruning and Knowledge Distillation (Minitron)* (2024) · Ma et al., *LLM-Pruner* (2023)
