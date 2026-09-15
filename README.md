# Améliorer un modèle d'IA sans le réentraîner de zéro

Panorama exécutable des méthodes pour améliorer un modèle existant : un dossier par méthode (ou groupe de méthodes qui se pratiquent de la même façon), avec pour chacun :

- un exemple de code aussi générique que possible ;
- des variantes quand un seul exemple ne couvre pas les cas courants ;
- le détail des types de modèles concernés ;
- la version « grosse infra » quand le poste local ne suffit pas.

Tous les exemples partagent **la même structure, la même ligne de commande, les mêmes données fil rouge et le même rapport avant/après**, pour passer facilement de l'un à l'autre (voir [docs/structure-commune.md](docs/structure-commune.md)).

> [!WARNING]
> **État : code écrit, pas encore exécuté.** Aucun exemple n'a encore tourné, pas même `--quick`. Les API de transformers 5.17, TRL 1.13 et PEFT 0.20 ont été vérifiées dans leur code source, mais des corrections sont à prévoir au premier lancement (`scripts/smoke.sh`).

## Démarrage

```bash
bash scripts/installer.sh            # venv + socle (transformers, trl, peft, accelerate, datasets...)
. .venv/bin/activate
python 02-fine-tuning-supervise/run.py --quick     # test rapide d'une méthode
python 02-fine-tuning-supervise/run.py             # exemple complet (GPU ~12 Go)
bash scripts/smoke.sh                               # tous les --quick, avec résumé
```

## Les méthodes

Légende de la colonne « Local 12 Go » : ✅ faisable sur un GPU grand public ; ⚠️ seulement sur de petits modèles ; ☁️ les vrais cas d'usage demandent une grosse infra (exemple fourni dans `infra/`).

### Entraîner les poids

| # | Méthode | Regroupe | Modèles concernés | Local 12 Go |
|---|---|---|---|---|
| [01](01-pre-entrainement-continu) | Pré-entraînement continu | Continued pretraining, DAPT, adaptation à une langue, MLM continu | LLM de base ; encodeurs (variante MLM) ; seq2seq, vision, audio (même principe) | ⚠️ ≤ 1,5B · ☁️ |
| [02](02-fine-tuning-supervise) | Fine-tuning supervisé | SFT, instruction tuning, fine-tuning complet | LLM ; variantes : encodeur (classification), vision (ViT), seq2seq (T5), speech (Whisper) | ⚠️ ≤ 1B (LLM), ✅ autres |
| [03](03-peft) | PEFT | LoRA, QLoRA, DoRA, rsLoRA, PiSSA, LoHa, LoKr, IA³, prompt et prefix tuning | Tous (couches linéaires) ; variante : diffusion (DreamBooth-LoRA) | ✅ ≤ 3B (LoRA), ≤ 8B (QLoRA) |

### Préférences et renforcement

| # | Méthode | Regroupe | Modèles concernés | Local 12 Go |
|---|---|---|---|---|
| [04](04-optimisation-preferences) | Optimisation des préférences | DPO, IPO, rDPO, SLiC, APO, NCA, BCO, SPPO, ORPO, SimPO, CPO, KTO ; variante DPO itératif | LLM (après SFT), VLM, diffusion (Diffusion-DPO) | ⚠️ ≤ 0,5B complet, ≤ 3B LoRA |
| [05](05-rl-en-ligne) | RL en ligne | RLHF, RLAIF, RLVR ; GRPO, Dr.GRPO, DAPO, GSPO, CISPO, RLOO ; PPO en infra ; variante modèle de récompense | LLM, VLM, agents ; diffusion (DDPO) | ⚠️ ≤ 1,5B · ☁️ |

### Transfert entre modèles

| # | Méthode | Regroupe | Modèles concernés | Local 12 Go |
|---|---|---|---|---|
| [06](06-distillation) | Distillation | Logits (KD), séquences, on-policy (GKD) ; variante classifieur encodeur | LLM (même tokenizer pour logits), encodeurs, vision, diffusion | ✅ petits couples · ☁️ 70B → 8B |
| [07](07-donnees-synthetiques) | Données synthétiques | Rejection sampling (RFT/STaR), Self-Instruct, critique-révision (Constitutional AI) | LLM ; données pour classifieurs, VLM, diffusion | ✅ ≤ 1B |
| [08](08-fusion-modeles) | Fusion de modèles | Model soups, SLERP, task arithmetic, TIES, DARE ; mergekit, fusion évolutionnaire | Fine-tunes d'une même base : LLM, LoRA, CLIP/ViT, diffusion | ✅ CPU/RAM |

### Interventions ciblées

| # | Méthode | Regroupe | Modèles concernés | Local 12 Go |
|---|---|---|---|---|
| [09](09-edition-connaissances) | Édition de connaissances | ROME, FT-L ; MEMIT, AlphaEdit en infra | LLM décodeurs ; diffusion (UCE, TIME) | ✅ ≤ 3B |
| [10](10-desapprentissage) | Désapprentissage | Gradient ascent, gradient difference, NPO, IDK ; TOFU, open-unlearning | LLM ; classifieurs ; diffusion (ESD) | ✅ ≤ 1B |
| [11](11-pilotage-activations) | Pilotage par activations | Steering vectors (ActAdd, CAA), abliteration | LLM décodeurs (tout flux résiduel) ; diffusion (concept sliders) | ✅ ≤ 7B |

### Modifier l'architecture

| # | Méthode | Regroupe | Modèles concernés | Local 12 Go |
|---|---|---|---|---|
| [12](12-extension-contexte) | Extension de contexte | Position Interpolation, NTK dynamique, YaRN, ABF | LLM à RoPE ; VLM (M-RoPE) ; encodeurs RoPE | ⚠️ petits modèles, 8k · ☁️ |
| [13](13-extension-vocabulaire) | Extension de vocabulaire | Added tokens, fusion BPE, init par moyenne, entraînement en 2 phases | LLM, encodeurs, seq2seq, tokens spéciaux VLM | ✅ ≤ 1B |
| [14](14-croissance-architecture) | Croissance d'architecture | Depth up-scaling (SOLAR), block expansion (LLaMA Pro), upcycling MoE | LLM famille LLaMA/Mistral ; encodeurs, ViT | ⚠️ transformation ✅, cicatrisation ☁️ |

### Compresser

| # | Méthode | Regroupe | Modèles concernés | Local 12 Go |
|---|---|---|---|---|
| [15](15-quantification) | Quantification | bitsandbytes, torchao, GPTQ, AWQ, QAT ; variante GGUF llama.cpp | Tous : LLM, MoE, encodeurs, VLM, diffusion, Whisper | ✅ |
| [16](16-elagage) | Élagage | Magnitude, Wanda, 2:4, suppression de couches, réduction de largeur + cicatrisation | LLM ; encodeurs ; CNN/ViT ; diffusion | ⚠️ ≤ 1,5B avec cicatrisation · ☁️ |

### Sans modifier les poids (aussi pour les modèles fermés via API)

| # | Méthode | Regroupe | Modèles concernés | Local 12 Go |
|---|---|---|---|---|
| [17](17-optimisation-prompt) | Optimisation du prompt | Instruction, CoT, few-shot (aléatoire / par similarité), OPRO ; DSPy en infra | Tout LLM local ou API ; VLM ; diffusion (réécriture de prompts) | ✅ |
| [18](18-rag) | RAG | BM25, dense, hybride RRF, reranker ; Qdrant + TEI + vLLM en infra | Tout LLM local ou API | ✅ |
| [19](19-outils-agents) | Outils et agents | Function calling, boucle agentique | LLM entraînés aux outils (≥ 3B), API | ✅ |
| [20](20-calcul-inference) | Calcul à l'inférence | Self-consistency, best-of-N (RM, vraisemblance), self-refine, pass@N | Tout LLM ; code ; diffusion (best-of-N) | ✅ |

## Documentation transverse

- [docs/choisir-une-methode.md](docs/choisir-une-methode.md) : quelle méthode pour quel besoin et quelle contrainte, et recette type d'un post-entraînement complet.
- [docs/structure-commune.md](docs/structure-commune.md) : organisation des dossiers, ligne de commande, clés de config, données fil rouge, socle `commun/`.
- [docs/oubli-catastrophique.md](docs/oubli-catastrophique.md) : le mesurer, le limiter (replay, LoRA, KL, fusion, EWC).

## Organisation

```
commun/                  socle partagé : config, modèles, données, évaluation, entraînement, générateur local/API, rapport
commun/infra/            accelerate FSDP2 / DeepSpeed ZeRO-3, Slurm multi-nœuds, évaluation lm-eval + vLLM
NN-methode/              README.md, run.py, config.yaml, [modules], [variantes/<nom>/], infra/
docs/                    documentation transverse
scripts/                 installer.sh, smoke.sh (tous les --quick)
sorties/                 résultats et modèles produits (ignoré par git)
```

## Environnement de référence

Python 3.13, PyTorch 2.14 (CUDA 13), transformers 5.17, TRL 1.13, PEFT 0.20, accelerate 1.15, datasets 5.0 ; poste avec RTX 5070 Ti Laptop (12 Go). Les dépendances propres à certaines méthodes sont listées dans [`requirements-extras.txt`](requirements-extras.txt).
