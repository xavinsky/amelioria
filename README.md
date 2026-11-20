# Améliorer un modèle d'IA sans le réentraîner de zéro

Site d'explication, avec l'arbre de décision et une page par méthode : **[AmeliorIA](https://tekio.org/amelioria/)**. Projet en cours ; licence MIT ([LICENSE](LICENSE)).

Panorama exécutable des méthodes pour améliorer un modèle existant : un dossier par méthode (ou groupe de méthodes qui se pratiquent de la même façon), avec pour chacun :

- un exemple de code aussi générique que possible ;
- des variantes quand un seul exemple ne couvre pas les cas courants ;
- le détail des types de modèles concernés ;
- la version « grosse infra » quand le poste local ne suffit pas.

Tous les exemples partagent **la même structure, la même ligne de commande, les mêmes données fil rouge et le même rapport avant/après**, pour passer facilement de l'un à l'autre (voir [docs/structure-commune.md](docs/structure-commune.md)).

> [!NOTE]
> **État.** Chaque exemple porte un statut honnête : 🟢 **Démontré** (gain mesuré et net), 🟡 **Non prouvé** (s'exécute, mais le résultat ne démontre pas encore le gain) ou ⚪ **Non testé** (versions grosse infra `infra/`, jamais exécutées). Tous les exemples locaux s'exécutent, en test rapide et en configuration complète, sur un GPU de 12 Go. Statuts, résultats et réserves : `make liste` ou [exemples.yaml](exemples.yaml) ; critères : [docs/cahier-des-charges.md](docs/cahier-des-charges.md).

## Démarrage

```bash
make install EXTRAS=1 CUDA=cu130     # venv .venv + socle + extras (bitsandbytes, torchao, diffusers, torchcodec)
make liste                           # les exemples par catégorie, avec statut et résultat mesuré
make test EX=FORM15                  # test rapide d'un exemple (petit modèle, quelques minutes au plus)
make run  EX=FORM15                  # exemple complet (GPU ~12 Go), résultats dans sorties/
```

`CUDA` choisit l'index PyTorch de la version CUDA du poste (`cu130`, `cu128`...) ; sans lui, torch vient de PyPI. Chaque commande vérifie d'abord les prérequis de l'exemple (modules, GPU, outils, API) et dit quoi installer s'il en manque un ; `make verifier EX=FORM15` fait seulement cette vérification.

| Option | Rôle | Exemple |
|---|---|---|
| `EX` | Méthode : identifiant (code de catégorie + numéro), numéro seul ou nom de dossier | `EX=DATA03`, `EX=03`, `EX=peft` |
| `VAR` | Variante du dossier | `make test EX=DATA03 VAR=diffusion-lora` |
| `SET` | Surcharges de config, séparées par des espaces | `SET="peft.methode=dora entrainement.lr=1e-4"` |
| `API`, `MODELE` | Modèle servi par une API compatible OpenAI (EXEC17 à EXEC20) | `API=http://localhost:11434/v1 MODELE=qwen3:8b` |
| `LANCEUR` | Préfixe de lancement | `LANCEUR="nice -n 10"` |

Dans un dossier de méthode, les mêmes cibles marchent sans `EX` : `cd 03-peft && make test VAR=diffusion-lora`. Autres cibles : `make install-gptq` (venv séparé pour GPTQ/AWQ), `make install-llamacpp` (variante GGUF), `make smoke` (tous les tests rapides), `make complet` (tous les exemples complets), `make graines` (répéter un exemple sur `N` graines : moyenne ± écart type), `make selecteurs` (chaque valeur des sélecteurs), `make help`.

## Les méthodes

<!-- methodes:debut (généré par make readme depuis exemples.yaml) -->

Statut de chaque exemple : 🟢 Démontré (gain mesuré et net) · 🟡 Non prouvé (s'exécute, gain pas encore démontré) · ⚪ Non testé (versions `infra/`, jamais exécutées). Résultats et réserves : `make liste`.

Colonne « Local 12 Go » : ✅ faisable sur un GPU grand public ; ⚠️ seulement sur de petits modèles ; ☁️ les vrais cas d'usage demandent une grosse infra (exemple fourni dans `infra/`).

### DATA · Apprendre à partir de données

Entraîner les poids sur des exemples, texte brut ou paires question/réponse.

| Id | Méthode | Regroupe | Modèles concernés | Local 12 Go | Exemples |
|---|---|---|---|---|---|
| [DATA01](01-pre-entrainement-continu) | Pré-entraînement continu | Continued pretraining, DAPT, adaptation à une langue, MLM continu | LLM de base ; encodeurs (variante MLM) ; seq2seq, vision, audio (même principe) | ⚠️ ≤ 1,5B · ☁️ | 🟡 principal · 🟢 mlm-encodeur |
| [DATA02](02-fine-tuning-supervise) | Fine-tuning supervisé | SFT, instruction tuning, fine-tuning complet | LLM ; variantes : encodeur (classification), vision (ViT), seq2seq (T5), speech (Whisper) | ⚠️ ≤ 1B (LLM), ✅ autres | 🟢 principal · 🟢 classification-encodeur · 🟢 vision · 🟢 seq2seq · 🟡 audio-whisper |
| [DATA03](03-peft) | PEFT (LoRA, QLoRA...) | LoRA, QLoRA, DoRA, rsLoRA, PiSSA, LoHa, LoKr, IA³, prompt et prefix tuning | Tous (couches linéaires) ; variante : diffusion (DreamBooth-LoRA) | ✅ ≤ 3B (LoRA), ≤ 8B (QLoRA) | 🟢 principal · 🟢 diffusion-lora |

### SIGN · Apprendre à partir d'un signal

Apprendre d'un jugement, d'une récompense, d'un enseignant ou de ses propres sorties filtrées.

| Id | Méthode | Regroupe | Modèles concernés | Local 12 Go | Exemples |
|---|---|---|---|---|---|
| [SIGN04](04-optimisation-preferences) | Optimisation des préférences | DPO, IPO, rDPO, SLiC, APO, NCA, BCO, SPPO, ORPO, SimPO, CPO, KTO ; variante DPO itératif | LLM (après SFT), VLM, diffusion (Diffusion-DPO) | ⚠️ ≤ 0,5B complet, ≤ 3B LoRA | 🟡 principal · 🟡 dpo-iteratif |
| [SIGN05](05-rl-en-ligne) | RL en ligne | RLHF, RLAIF, RLVR ; GRPO, Dr.GRPO, DAPO, GSPO, CISPO, RLOO ; PPO en infra ; variante modèle de récompense | LLM, VLM, agents ; diffusion (DDPO) | ⚠️ ≤ 1,5B · ☁️ | 🟢 principal · 🟢 modele-recompense |
| [SIGN06](06-distillation) | Distillation | Logits (KD), séquences, on-policy (GKD) ; variante classifieur encodeur | LLM (même tokenizer pour logits), encodeurs, vision, diffusion | ✅ petits couples · ☁️ 70B → 8B | 🟢 principal · 🟢 encodeur-classification |
| [SIGN07](07-donnees-synthetiques) | Données synthétiques | Rejection sampling (RFT/STaR), Self-Instruct, critique-révision (Constitutional AI) | LLM ; données pour classifieurs, VLM, diffusion | ✅ ≤ 1B | 🟡 principal |

### EDIT · Chirurgie du modèle

Combiner des modèles, corriger, effacer ou orienter un savoir ou un comportement précis.

| Id | Méthode | Regroupe | Modèles concernés | Local 12 Go | Exemples |
|---|---|---|---|---|---|
| [EDIT08](08-fusion-modeles) | Fusion de modèles | Model soups, SLERP, task arithmetic, TIES, DARE ; mergekit, fusion évolutionnaire | Fine-tunes d'une même base : LLM, LoRA, CLIP/ViT, diffusion | ✅ CPU/RAM | 🟢 principal |
| [EDIT09](09-edition-connaissances) | Édition de connaissances | ROME, FT-L ; MEMIT, AlphaEdit en infra | LLM décodeurs ; diffusion (UCE, TIME) | ✅ ≤ 3B | 🟢 principal |
| [EDIT10](10-desapprentissage) | Désapprentissage | Gradient ascent, gradient difference, NPO, IDK ; TOFU, open-unlearning | LLM ; classifieurs ; diffusion (ESD) | ✅ ≤ 1B | 🟢 principal |
| [EDIT11](11-pilotage-activations) | Pilotage par activations | Steering vectors (ActAdd, CAA), abliteration | LLM décodeurs (tout flux résiduel) ; diffusion (concept sliders) | ✅ ≤ 7B | 🟢 principal |

### FORM · Remodeler : étendre ou compresser

Changer la forme du modèle (fenêtre de contexte, vocabulaire, couches, précision, taille).

| Id | Méthode | Regroupe | Modèles concernés | Local 12 Go | Exemples |
|---|---|---|---|---|---|
| [FORM12](12-extension-contexte) | Extension de contexte | Position Interpolation, NTK dynamique, YaRN, ABF | LLM à RoPE ; VLM (M-RoPE) ; encodeurs RoPE | ⚠️ petits modèles, 8k · ☁️ | 🟢 principal |
| [FORM13](13-extension-vocabulaire) | Extension de vocabulaire | Added tokens, fusion BPE, init par moyenne, entraînement en 2 phases | LLM, encodeurs, seq2seq, tokens spéciaux VLM | ✅ ≤ 1B | 🟢 principal |
| [FORM14](14-croissance-architecture) | Croissance d'architecture | Depth up-scaling (SOLAR), block expansion (LLaMA Pro), upcycling MoE | LLM famille LLaMA/Mistral ; encodeurs, ViT | ⚠️ transformation ✅, cicatrisation ☁️ | 🟢 principal |
| [FORM15](15-quantification) | Quantification | bitsandbytes, torchao, GPTQ, AWQ, QAT ; variante GGUF llama.cpp | Tous : LLM, MoE, encodeurs, VLM, diffusion, Whisper | ✅ | 🟢 principal · 🟢 gguf-llamacpp |
| [FORM16](16-elagage) | Élagage | Magnitude, Wanda, 2:4, suppression de couches, réduction de largeur + cicatrisation | LLM ; encodeurs ; CNN/ViT ; diffusion | ⚠️ ≤ 0,5B avec cicatrisation · ☁️ | 🟡 principal |

### EXEC · À l'exécution, sans toucher aux poids

Améliorer le résultat autour du modèle ; fonctionne aussi avec un modèle fermé via API.

| Id | Méthode | Regroupe | Modèles concernés | Local 12 Go | Exemples |
|---|---|---|---|---|---|
| [EXEC17](17-optimisation-prompt) | Optimisation du prompt | Instruction, CoT, few-shot (aléatoire / par similarité), OPRO ; DSPy en infra | Tout LLM local ou API ; VLM ; diffusion (réécriture de prompts) | ✅ | 🟢 principal |
| [EXEC18](18-rag) | RAG | BM25, dense, hybride RRF, reranker ; Qdrant + TEI + vLLM en infra | Tout LLM local ou API | ✅ | 🟢 principal |
| [EXEC19](19-outils-agents) | Outils et agents | Function calling, boucle agentique | LLM entraînés aux outils (≥ 3B), API | ✅ | 🟢 principal |
| [EXEC20](20-calcul-inference) | Calcul à l'inférence | Self-consistency, best-of-N (RM, vraisemblance), self-refine, pass@N | Tout LLM ; code ; diffusion (best-of-N) | ✅ | 🟢 principal |

<!-- methodes:fin -->

## Documentation transverse

- [docs/cahier-des-charges.md](docs/cahier-des-charges.md) : ce que chaque exemple doit démontrer (besoin réel, bon concept, preuve sans ambiguïté).
- [docs/transformers.md](docs/transformers.md) : comment fonctionne un transformer (schéma principal, entraînement, génération, familles de modèles).
- [docs/schemas-methodes.md](docs/schemas-methodes.md) : un schéma par méthode et par variante, montrant ce qui est gelé, entraîné, ajouté ou modifié.
- [docs/bibliotheques.md](docs/bibliotheques.md) : les bibliothèques Python utilisées, avec les classes et fonctions appelées.
- [docs/references.md](docs/references.md) : les papiers de recherche de chaque technique, avec leurs liens arXiv.
- [docs/choisir-une-methode.md](docs/choisir-une-methode.md) : quelle méthode pour quel besoin et quelle contrainte, et recette type d'un post-entraînement complet.
- [docs/structure-commune.md](docs/structure-commune.md) : organisation des dossiers, ligne de commande, clés de config, données fil rouge, socle `commun/`.
- [docs/oubli-catastrophique.md](docs/oubli-catastrophique.md) : le mesurer, le limiter (replay, LoRA, KL, fusion, EWC).

## Organisation

```
commun/                  socle partagé : config, modèles, données, évaluation, preuve, entraînement, générateur local/API, rapport
commun/infra/            accelerate FSDP2 / DeepSpeed ZeRO-3, Slurm multi-nœuds, évaluation lm-eval + vLLM
NN-methode/              README.md, Makefile, run.py, config.yaml, [modules], [variantes/<nom>/], infra/
docs/                    documentation transverse
Makefile, exemples.yaml  commandes communes ; catalogue (catégories, statuts, résultats, prérequis)
scripts/                 catalogue.py (liste, vérification, section du README), installer.sh, smoke.sh, selecteurs.sh
sorties/, logs/          résultats, modèles produits et journaux (ignorés par git)
backlog.md, TODONEXT.md, session-status.md   état des exemples, prochaines étapes, avancement et notes de session
```

## Environnement de référence

Python 3.13, PyTorch 2.14 (CUDA 13), transformers 5.17, TRL 1.13, PEFT 0.20, accelerate 1.15, datasets 5.0, sentence-transformers 6.0 ; extras : bitsandbytes 0.50, torchao 0.18, diffusers 0.40, torchcodec 0.16 ; llm-compressor 0.13 dans un venv séparé. Poste de test : RTX 5070 Ti Laptop (12 Go, 11,5 Gio utilisables), 30 Go de RAM. Les dépendances propres à certaines méthodes sont listées dans [`requirements-extras.txt`](requirements-extras.txt).
