# 03 — Fine-tuning efficace en paramètres (PEFT : LoRA, QLoRA, DoRA...)

> Geler le modèle et n'entraîner qu'un petit module ajouté : 0,01 à 2 % des paramètres, pour une qualité proche du fine-tuning complet.

| Regroupe | Modifie | Local 12 Go | Test `--quick` |
|---|---|---|---|
| LoRA, QLoRA, DoRA, rsLoRA, PiSSA, LoHa, LoKr, IA³, prompt tuning, prefix tuning, DreamBooth-LoRA | Un adaptateur ajouté (poids d'origine intacts) | ✅ LoRA jusqu'à ~3B, QLoRA jusqu'à ~8B | ✅ |

## Principe

Schéma : [docs/schemas-methodes.md](../docs/schemas-methodes.md#m03) (ce que la méthode ajoute ou modifie dans le transformer).

Toutes ces méthodes suivent la même recette : mêmes données et même boucle que le SFT ([DATA02](../02-fine-tuning-supervise)). Seul change le petit module entraîné, d'où leur regroupement ici. On passe de l'une à l'autre avec `--set peft.methode=...`.

| Méthode | Ce qui est entraîné | Particularité |
|---|---|---|
| **LoRA** | Deux petites matrices `B·A` de rang `r` ajoutées à chaque couche linéaire : `W' = W + (α/r)·B·A` | Standard de fait. Adaptateur de quelques Mo, fusionnable dans les poids |
| **QLoRA** | LoRA sur un modèle chargé en 4 bits (NF4) | Divise la mémoire par ~4 : un 8B tient sur 12 Go, un 70B sur 1 GPU 80 Go |
| **DoRA** | LoRA sur la *direction* des poids, plus une magnitude apprise séparément | Souvent un peu meilleur que LoRA à rang égal, un peu plus lent |
| **rsLoRA** | LoRA avec échelle `α/√r` | Stable quand on monte le rang (64 à 256) |
| **PiSSA** | LoRA initialisée sur les valeurs singulières principales de `W` | Converge plus vite |
| **LoHa / LoKr** | Produit de Hadamard ou de Kronecker au lieu de `B·A` | Populaires en diffusion (LyCORIS) |
| **IA³** | Vecteurs qui multiplient les activations (clés, valeurs, FFN) | ~0,01 % de paramètres, lr élevé |
| **Prompt tuning** | Embeddings de « tokens virtuels » ajoutés au début du prompt | Le plus léger. Efficace surtout sur les gros modèles |
| **Prefix tuning** | Tokens virtuels injectés dans chaque couche (clés/valeurs) | Plus expressif que le prompt tuning |

Intérêts pratiques :

- plusieurs adaptateurs pour un seul modèle de base, chargés à chaud (vLLM `--enable-lora`, TGI, LoRAX) ;
- combinaison d'adaptateurs ([EDIT08](../08-fusion-modeles)) ;
- moins d'oubli que le fine-tuning complet (*LoRA Learns Less and Forgets Less*).

## Modèles concernés

| Type de modèle | Exemples de modèles | Applicable | Remarques |
|---|---|---|---|
| LLM décodeur (toutes familles HF) | Llama 3, Qwen2.5, Mistral 7B, Gemma 2 | ✅ cas principal | [`run.py`](run.py). `all-linear` cible toutes les couches linéaires, quelle que soit l'architecture |
| LLM quantifié (bitsandbytes 4/8 bits, GPTQ, AWQ) | Qwen2.5-7B-Instruct-AWQ, Llama-3.1-8B en NF4 | ✅ | QLoRA. Entraîner une LoRA sur un modèle GPTQ/AWQ est aussi possible |
| Encodeur / seq2seq / Whisper | BERT, T5, Whisper | ✅ | Même API, `task_type="SEQ_CLS"`, `"SEQ_2_SEQ_LM"`... |
| Vision / multimodal | ViT, CLIP, Qwen2.5-VL, LLaVA | ✅ | Cibler les couches linéaires de la tour vision et/ou du LLM |
| Diffusion | Stable Diffusion, SDXL, Flux | ✅ très courant | [`variantes/diffusion-lora`](variantes/diffusion-lora) : apprendre un sujet ou un style en quelques images |
| Tout `nn.Module` PyTorch personnalisé | — | ✅ | PEFT accepte des modèles quelconques via `target_modules` |
| Modèle fermé via API | GPT-4o, Claude, Gemini | ⚠️ | Certains fournisseurs font du LoRA côté serveur (fine-tuning « as a service ») |

## Quand l'utiliser

- ✅ Par défaut dès qu'on veut fine-tuner : c'est moins cher, plus rapide et le résultat se déploie facilement.
- ✅ QLoRA dès que le modèle en bf16 ne tient pas en mémoire.
- ✅ Beaucoup de variantes spécialisées d'un même modèle (un adaptateur par client, par tâche).
- ⚠️ Pour injecter beaucoup de connaissances nouvelles ou faire un gros changement de distribution, le fine-tuning complet reste supérieur. Une LoRA de rang élevé (128 à 256) sur toutes les couches s'en rapproche.

## Composants de l'exemple

```mermaid
flowchart TD
  classDef externe fill:#fed7aa,stroke:#ea580c,color:#7c2d12
  classDef code fill:#ffffff,stroke:#6b7280,color:#1f2937
  classDef gpu fill:#fecaca,stroke:#dc2626,color:#7f1d1d
  classDef disque fill:#dcfce7,stroke:#16a34a,color:#14532d

  Hub["Hugging Face Hub<br/>Qwen2.5-0.5B-Instruct"]
  Cache[("cache disque<br/>~/.cache/huggingface")]
  subgraph Py["Processus Python"]
    Run["run.py<br/>charge les données<br/>et le modèle"]
    Ev["evaluation.py<br/>génération"]
    TRL["TRL SFTTrainer<br/>LoRA r = 16"]
  end
  GPU["GPU CUDA, 12 Go"]
  Out[("sorties/03-peft/")]

  Hub -- "① HTTPS GET, 1re fois<br/>poids (1 Go), banking77, Alpaca" --> Cache
  Cache -- "② lecture des poids" --> Run
  Run -- "③ copie des poids<br/>en mémoire GPU" --> GPU
  Run -- "④ ⑨ appel Python<br/>mesure avant, après" --> Ev
  Ev -- "⑤ ⑩ noyaux CUDA<br/>génération" --> GPU
  Run -- "⑥ appel Python<br/>trainer.train()" --> TRL
  TRL -- "⑦ noyaux CUDA<br/>forward, backward" --> GPU
  TRL -- "⑧ écriture<br/>adaptateur" --> Out
  Ev -- "⑪ écriture<br/>resultats.json" --> Out

  class Hub externe
  class Run,Ev,TRL code
  class GPU gpu
  class Cache,Out disque
```

`make run EX=DATA03` vérifie les prérequis (modules, GPU), puis lance `run.py` dans l'environnement `.venv`. Le script lit `config.yaml` (modèle, méthode PEFT, données, réglages). Données, remélange et évaluation sont exactement ceux du fine-tuning complet ([DATA02](../02-fine-tuning-supervise)) : router les messages de clients d'une banque vers l'une de 77 intentions (banking77), avec la liste des intentions en message système.

- **① Téléchargement.** Au premier lancement, les poids de Qwen2.5-0.5B-Instruct (environ 1 Go), le jeu banking77 et les demandes générales d'Alpaca sont téléchargés depuis le Hugging Face Hub, par HTTPS.
- **② Lecture.** Tout est rangé dans le cache `~/.cache/huggingface` ; les lancements suivants le relisent sur le disque, sans réseau.
- **③ Copie sur le GPU.** Les poids sont copiés en bf16 dans la mémoire du GPU, où ils restent jusqu'à la fin. En QLoRA (`SET="peft.methode=qlora"`), ils y sont stockés en 4 bits par bitsandbytes.
- **④ Mesure avant.** `run.py` appelle `commun/evaluation.py`, qui pose au modèle les 3 080 messages de test, par lots de 32, puis les 200 problèmes d'arithmétique de non-régression.
- **⑤ Génération sur le GPU.** Au plus 16 tokens par message, dont la première ligne doit être l'intention attendue ; jusqu'à 256 tokens de raisonnement par problème d'arithmétique.
- **⑥ Entraînement.** Le modèle répond d'abord aux demandes générales du remélange (ou les relit dans `sorties/cache-remelange/`), puis `run.py` confie modèle, exemples et configuration LoRA au SFTTrainer de TRL. PEFT ajoute deux petites matrices de rang 16 à chaque couche linéaire et gèle tout le reste : 8,8 millions de paramètres entraînés sur 503 millions (1,7 %).
- **⑦ Forward et backward sur le GPU.** Chaque pas traite 16 exemples (lots de 8, accumulés 2 fois) : passe avant, perte sur la réponse seulement, passe arrière, mise à jour des seules matrices LoRA. 2 époques, taux d'apprentissage 2e-4.
- **⑧ Écriture de l'adaptateur** dans `sorties/03-peft/adaptateur-lora/`.
- **⑨ Mesure après**, sur les mêmes messages et problèmes, adaptateur actif, puis la non-régression une seconde fois, adaptateur désactivé.
- **⑩ Génération sur le GPU**, avec le même décodage qu'en ⑤.
- **⑪ Écriture du rapport** dans `sorties/03-peft/resultats.json`, avec le même tableau affiché dans le terminal.

**Ce que fait le GPU.** Le processus Python ne calcule presque rien lui-même : PyTorch envoie au GPU des « noyaux CUDA », de petits programmes qui font les multiplications de matrices et l'attention sur les poids et les activations déjà présents dans sa mémoire. En génération (⑤, ⑩), il y a une passe avant par token produit, accélérée par un cache des clés et des valeurs déjà calculées. En entraînement (⑦), chaque pas fait une passe avant, une passe arrière qui calcule les gradients, puis la mise à jour des paramètres LoRA par l'optimiseur AdamW, dont l'état n'existe que pour ces paramètres. Le pic mesuré est de 8,7 Gio de mémoire GPU, presque autant qu'en fine-tuning complet (9,2 Gio) sur un modèle aussi petit : les activations et les logits d'un vocabulaire de 152 000 tokens dominent. L'écart se creuse avec la taille du modèle, car poids fp32, gradients et état d'AdamW n'existent plus que pour l'adaptateur.

**Ce qui est écrit.**

| Fichier | Contenu |
|---|---|
| `adaptateur-lora/adapter_model.safetensors` | Les matrices LoRA entraînées (35 Mo, en fp32), seules à être sauvegardées : le modèle de base reste celui du Hub |
| `adaptateur-lora/adapter_config.json` | Rang, alpha, couches ciblées (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`) et nom du modèle de base, pour recharger l'adaptateur avec `PeftModel.from_pretrained` |
| `adaptateur-lora/tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja` | Le tokenizer et le format de conversation, copiés pour que le dossier se suffise à lui-même |
| `adaptateur-lora/training_args.bin` | Les réglages d'entraînement utilisés |
| `resultats.json` | Exactitude avant, après et adaptateur désactivé, avec intervalle à 95 % et réponse de chaque question ; comparaisons appariées (différence, intervalle, p de McNemar) ; paramètres entraînés, taille de l'adaptateur, durée d'entraînement et pic de mémoire GPU |
| `config_effective.yaml` | La configuration réellement utilisée, surcharges `SET` comprises |

Avec `SET="peft.fusionner=true"`, le dossier `modele-fusionne/` contient en plus le modèle complet avec l'adaptateur intégré, prêt à servir sans PEFT.

## Utilisation

```bash
python run.py --quick                                        # LoRA sur SmolLM2-135M
python run.py                                                # LoRA r=16 sur Qwen2.5-0.5B-Instruct, banking77, ~30 min
python run.py --set donnees.remelange.part=0                 # sans remélange
python run.py --set peft.methode=qlora --set modele.nom=Qwen/Qwen2.5-7B-Instruct --set entrainement.gradient_checkpointing=true
python run.py --set peft.methode=dora
python run.py --set peft.methode=prompt_tuning
python run.py --set peft.fusionner=true                      # écrit aussi le modèle fusionné
for m in lora dora rslora pissa loha lokr ia3 prompt_tuning prefix_tuning; do python run.py --quick --set peft.methode=$m; done
python variantes/diffusion-lora/run.py --quick               # pipeline SD miniature
python variantes/diffusion-lora/run.py                       # SD 1.5, 5 photos de chien, ~10 min
```

## Ce que fait l'exemple

Même besoin, mêmes données et même évaluation que le fine-tuning complet ([DATA02](../02-fine-tuning-supervise)) : router les messages de clients d'une banque vers l'une de 77 intentions. Seule change la part des poids entraînée.

**Résultats** (RTX 5070 Ti Laptop, 12 Go ; 3 080 messages de test, 200 problèmes de non-régression) :

| | Intentions | Non-régression (arithmétique) | Entraîné | Entraînement |
|---|---|---|---|---|
| Qwen2.5-0.5B-Instruct, avant | 0,229 [0,214 ; 0,244] | 0,955 [0,917 ; 0,976] | | |
| **LoRA, avec remélange** (`config.yaml`) | **0,891** [0,879 ; 0,901] | **0,95** [0,910 ; 0,973] | 1,7 %, adaptateur de 35 Mo | 18 min, 8,7 Gio |
| même modèle, adaptateur désactivé | — | 0,955 | | |
| LoRA, sans remélange | 0,925 [0,915 ; 0,934] | 0,33 [0,269 ; 0,398] | | 14 min, 8,7 Gio |
| SFT complet, avec remélange ([DATA02](../02-fine-tuning-supervise)) | 0,904 [0,893 ; 0,914] | 0,765 [0,702 ; 0,818] | 100 %, modèle de 2 Go | 23 min, 9,2 Gio |

Gain sur les intentions, apparié sur les mêmes messages : +0,662 [+0,645 ; +0,679], p de McNemar < 10⁻³⁰⁰. Non-régression : −0,005 [−0,035 ; +0,025], p = 1,0, aucun écart mesurable.

**Ce qu'on en retient.**

- **LoRA apprend presque autant que le fine-tuning complet** : 0,891 contre 0,904, en n'entraînant que 1,7 % des paramètres.
- **Et oublie beaucoup moins** : avec le même remélange, l'arithmétique reste à 0,95, contre 0,765 en fine-tuning complet. C'est le résultat de Biderman et al. (2024), *LoRA Learns Less and Forgets Less*.
- **L'adaptateur se désactive** : sans lui, le modèle est exactement celui d'origine (0,955). Un seul modèle de base peut servir plusieurs tâches, chacune avec son adaptateur de quelques dizaines de Mo.
- **Le remélange reste nécessaire** : sans lui, même LoRA perd le raisonnement pas à pas (0,33), car il apprend à répondre par un mot.

## Paramètres principaux (`config.yaml`)

| Clé | Rôle |
|---|---|
| `peft.methode` | Voir le tableau ci-dessus |
| `peft.r`, `peft.alpha` | Rang et échelle LoRA (`alpha = 2r` est un bon départ) |
| `peft.modules` | `all-linear` (recommandé) ou une liste de noms de couches |
| `entrainement.lr` / `lr_par_methode` | LoRA ~1e-4 à 2e-4 ; IA³ et prompt tuning beaucoup plus haut |
| `peft.fusionner` | `merge_and_unload()` puis sauvegarde du modèle complet |

## Grosse infra

- En local : LoRA bf16 jusqu'à ~3B, QLoRA jusqu'à ~8B sur 12 Go.
- [`infra/config.yaml`](infra/config.yaml) : **QLoRA sur Llama-3.3-70B**, FSDP sur 2 à 8 GPU 80 Go. Le 4 bits est découpé grâce à `bnb_4bit_quant_storage`, réglé automatiquement dans `commun/modeles.py`.
- [`infra/lancer.sh`](infra/lancer.sh) : entraînement, puis service de plusieurs adaptateurs sur une même base avec vLLM (`--enable-lora --lora-modules`).
- Diffusion à grande échelle : scripts officiels diffusers `train_dreambooth_lora_sdxl.py`, `train_dreambooth_lora_flux.py` (Flux.1-dev : 24 à 40 Go).

## Pièges

- **lr** : un lr de fine-tuning complet (1e-5) fait stagner la LoRA ; un lr de LoRA (2e-4) fait diverger le fine-tuning complet.
- **Modules ciblés** : une LoRA sur l'attention seule est nettement moins bonne que `all-linear`.
- **QLoRA et fusion** : ne pas fusionner dans le modèle 4 bits. Recharger la base en bf16, fusionner, puis requantifier si besoin.
- **Prompt et prefix tuning** : faibles sur les petits modèles (< 1B). C'est attendu, pas un bug.
- **Prefix tuning et gradient checkpointing** : incompatibles (le checkpointing retire le cache où sont injectés les préfixes). `run.py` le désactive pour `prefix_tuning`.
- **Diffusion** : trop de pas donne un surapprentissage (toutes les images se ressemblent). Régulariser avec des « images de classe » (prior preservation).

## Références

- Hu et al., *LoRA* (2021) · Dettmers et al., *QLoRA* (2023) · Liu et al., *DoRA* (2024) · Kalajdzievski, *rsLoRA* (2023) · Meng et al., *PiSSA* (2024)
- Liu et al., *IA³ / T-Few* (2022) · Lester et al., *Prompt Tuning* (2021) · Li & Liang, *Prefix-Tuning* (2021)
- Biderman et al., *LoRA Learns Less and Forgets Less* (2024) · Ruiz et al., *DreamBooth* (2022)
- Casanueva et al., *Efficient Intent Detection with Dual Sentence Encoders* (2020) : le jeu banking77 ([2003.04807](https://arxiv.org/abs/2003.04807))
- https://huggingface.co/docs/peft
