# 01 — Pré-entraînement continu

> Poursuivre l'entraînement « prédire le token suivant » d'un modèle déjà pré-entraîné, sur un corpus brut du domaine ou de la langue visés.

| Regroupe | Modifie | Local 12 Go | Test `--quick` |
|---|---|---|---|
| Continued pretraining, DAPT (domain-adaptive pretraining), adaptation à une langue, MLM continu des encodeurs | Tous les poids | ✅ jusqu'à ~1,5B (complet), ❌ au-delà | ✅ SmolLM2-135M |

## Principe

Schéma : [docs/schemas-methodes.md](../docs/schemas-methodes.md#m01) (ce que la méthode ajoute ou modifie dans le transformer).

On reprend exactement l'objectif du pré-entraînement initial, sans questions ni réponses : du texte brut du domaine visé. Le modèle absorbe le vocabulaire, le style et les connaissances du corpus : textes médicaux, juridiques, code interne, une langue peu représentée.

Trois réglages font la différence :

- **Learning rate faible** : 10 à 100 fois plus faible qu'au pré-entraînement initial, avec un warmup puis une décroissance cosinus.
- **Replay** : on mélange 5 à 30 % de texte général au corpus du domaine. Sans ça, l'oubli catastrophique est quasi certain (voir [docs/oubli-catastrophique.md](../docs/oubli-catastrophique.md)).
- **Diversité des formulations** : un fait vu sous une seule formulation est mémorisé mais rarement réutilisable en question-réponse. D'où plusieurs paraphrases par fait dans l'exemple.

Le pré-entraînement continu est en général suivi d'un SFT ([02](../02-fine-tuning-supervise)) pour récupérer le comportement d'assistant.

## Modèles concernés

| Type de modèle | Applicable | Remarques |
|---|---|---|
| LLM décodeur de base (LLaMA, Qwen, Mistral, Gemma, Phi, GPT-2, SmolLM...) | ✅ cas principal | `run.py`, tout `AutoModelForCausalLM` |
| LLM instruct / chat | ⚠️ | Fonctionne, mais le format de chat se dégrade. Préférer : base -> pré-entraînement continu -> SFT, ou fusionner ensuite avec l'instruct ([08](../08-fusion-modeles)) |
| Encodeur (BERT, RoBERTa, DeBERTa, CamemBERT) | ✅ | [`variantes/mlm-encodeur`](variantes/mlm-encodeur) : objectif MLM |
| Encodeur-décodeur (T5, BART) | ✅ | Même idée avec l'objectif de débruitage d'origine (span corruption) |
| Vision (ViT, DINOv2) / audio (wav2vec2) | ✅ | Objectifs auto-supervisés d'origine (MAE, DINO, contrastif) sur images ou sons du domaine |
| Diffusion | ⚠️ | Équivaut à un fine-tuning complet sur les images du domaine (voir [03 variante diffusion](../03-peft/variantes/diffusion-lora)) |
| Modèle fermé via API | ❌ | Pas d'accès aux poids |

## Quand l'utiliser

- ✅ Le domaine a un vocabulaire et une distribution de texte très différents du web (biomédical, juridique, langue rare, code propriétaire).
- ✅ Vous avez beaucoup de texte brut non annoté : de 10 M à plusieurs dizaines de milliards de tokens.
- ❌ Pour des connaissances qui changent souvent ou doivent être citées : préférer le RAG ([18](../18-rag)).
- ❌ Pour changer un format ou un comportement : préférer le SFT ([02](../02-fine-tuning-supervise)) ou LoRA ([03](../03-peft)).

## Utilisation

```bash
python run.py --quick                                  # quelques secondes, SmolLM2-135M
python run.py                                          # Qwen2.5-0.5B, 40 personnages fictifs, ~5 min sur GPU 12 Go
python run.py --set donnees.replay_ratio=0             # sans replay : observer la hausse de perplexite_generale
python run.py --set modele.nom=Qwen/Qwen2.5-1.5B --set entrainement.gradient_checkpointing=true
python variantes/mlm-encodeur/run.py --quick           # encodeur RoBERTa
```

## Ce que fait l'exemple

1. Génère une base de 40 personnages fictifs (4 formulations par personnage) que le modèle ne peut pas connaître.
2. Mesure avant l'entraînement : la perplexité sur la 4e formulation jamais vue, l'exactitude en question-réponse sur ces personnages et la perplexité wikitext (connaissances générales).
3. Entraîne sur les 3 autres formulations, avec 20 % de replay wikitext.
4. Refait les mesures. Attendu : la perplexité du domaine chute, l'exactitude QA monte et la perplexité générale reste stable.

## Paramètres principaux (`config.yaml`)

| Clé | Rôle |
|---|---|
| `modele.nom` | N'importe quel modèle causal HF |
| `donnees.source` | `faits` (exemple fictif) ou `hf` (corpus réel : `nom`, `config`, `colonne`) |
| `donnees.replay_ratio` | Part de texte général mélangé |
| `entrainement.longueur_bloc` | Taille des séquences concaténées |
| `entrainement.lr`, `epochs` | Voir *Principe* |

## Grosse infra

Pré-entraînement continu impossible en local au-delà de ~1,5B de paramètres en entraînement complet. Adam en bf16 demande ~16 octets par paramètre, sans compter les activations.

- [`infra/config.yaml`](infra/config.yaml) : Llama-3.1-8B, français (FineWeb-2), séquences de 8k, FSDP2.
- [`infra/lancer.sh`](infra/lancer.sh) : 1 nœud de 8 GPU ; multi-nœuds via [`commun/infra/slurm_multinoeud.sbatch`](../commun/infra/slurm_multinoeud.sbatch).
- Ordres de grandeur : 8B sur 8×H100 ≈ 8 000 tokens/s/GPU, soit ~1,5 jour pour 10 Md de tokens. 70B demande 4 à 8 nœuds.
- Au-delà de quelques milliards de tokens, charger le corpus en mémoire n'est plus possible. On passe à un pipeline dédié :
  - [torchtitan](https://github.com/pytorch/torchtitan) ou [Nanotron](https://github.com/huggingface/nanotron) : FSDP2 + parallélisme tensoriel et pipeline, données tokenisées à l'avance, en streaming ;
  - [NeMo / Megatron-LM](https://github.com/NVIDIA/NeMo) sur cluster NVIDIA.

  Les réglages (lr, replay, warmup) restent ceux de cet exemple.

## Pièges

- **Oubli catastrophique** : suivre `perplexite_generale`, garder du replay.
- **Tokenizer inadapté** : une langue ou un domaine mal découpés en tokens coûtent cher et apprennent mal. Voir l'extension de vocabulaire ([13](../13-extension-vocabulaire)).
- **Re-warmup** : repartir avec le lr maximal provoque un pic de perte. Un warmup court suffit.
- **Qualité des données** : dédupliquer et filtrer le corpus compte plus que sa taille.

## Références

- Gururangan et al., *Don't Stop Pretraining* (2020) : DAPT/TAPT.
- Ibrahim et al., *Simple and Scalable Strategies to Continually Pre-train LLMs* (2024) : re-warming et replay.
- Allen-Zhu & Li, *Physics of Language Models 3.1* (2023) : augmentation par paraphrases pour l'extraction de connaissances.
