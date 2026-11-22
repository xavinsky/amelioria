# 05 — RL en ligne (RLHF, RLAIF, RLVR : GRPO, DAPO, GSPO, RLOO...)

> Le modèle génère plusieurs réponses, une récompense les note, et on renforce celles qui ont mieux marché que la moyenne du groupe.

| Regroupe | Modifie | Local 12 Go | Test `--quick` |
|---|---|---|---|
| RLHF (modèle de récompense), RLAIF / Constitutional AI (LLM juge), RLVR (récompense vérifiable) ; GRPO, Dr.GRPO, DAPO, GSPO, CISPO, RLOO ; PPO en infra | Tous les poids (ou LoRA) | ✅ jusqu'à ~1,5B (lent : la génération domine) | ✅ |

## Principe

Schéma : [docs/schemas-methodes.md](../docs/schemas-methodes.md#m05) (ce que la méthode ajoute ou modifie dans le transformer).

À chaque pas :

1. On tire un lot de prompts.
2. Le modèle génère **G réponses par prompt** (échantillonnage à température 1).
3. Chaque réponse reçoit une **récompense**.
4. L'**avantage** de chaque réponse est sa récompense moins la moyenne de son groupe (éventuellement divisée par l'écart-type). On n'a donc pas besoin d'un réseau critique comme dans PPO.
5. Mise à jour de type « policy gradient » avec clipping, et éventuellement une pénalité KL vers le modèle initial.

Contrairement au DPO ([SIGN04](../04-optimisation-preferences)), les données sont **générées par le modèle lui-même** pendant l'entraînement (on-policy). Le modèle peut ainsi découvrir des stratégies meilleures que celles des données d'origine. C'est ce qui a produit les modèles de raisonnement (o1, DeepSeek-R1).

**La boucle est la même dans tous les cas, seule la source de récompense change**, d'où le regroupement :

| `recompense.source` | Nom usuel | Récompense | Idéal pour |
|---|---|---|---|
| `regles` | **RLVR** | Programme vérificateur : réponse exacte, tests unitaires qui passent, format respecté | Maths, code, extraction structurée, tout ce qui se vérifie |
| `modele` | **RLHF** | Modèle de récompense entraîné sur des préférences humaines ([`variantes/modele-recompense`](variantes/modele-recompense)) | Utilité, ton, sécurité : ce qui ne se vérifie pas |
| `juge` | **RLAIF** / Constitutional AI | Un LLM note la réponse selon une consigne ou des principes écrits | Idem, sans annotateurs humains |

Et les variantes d'algorithme (`--set algo=...`) :

| Algo | Différence |
|---|---|
| `grpo` | Formulation DeepSeekMath : avantages normalisés par groupe, KL β=0,04 |
| `dr_grpo` | Supprime les biais de longueur et de difficulté (pas de division par l'écart-type) |
| `dapo` | Clipping asymétrique (clip-higher), normalisation par token, masque des réponses tronquées. Défaut TRL |
| `gspo` | Ratio d'importance au niveau de la séquence : plus stable, surtout pour les MoE (Qwen3) |
| `cispo` | Clipping des poids d'importance plutôt que des mises à jour (MiniMax-M1) |
| `rloo` | REINFORCE leave-one-out : baseline = moyenne des autres réponses |
| PPO | Avec critique appris. Plus coûteux, encore utilisé à grande échelle : voir l'infra (verl, OpenRLHF) |

## Modèles concernés

| Type de modèle | Exemples de modèles | Applicable | Remarques |
|---|---|---|---|
| LLM décodeur instruct ou après SFT | Llama 3, Qwen2.5, Mistral 7B, Gemma 2 | ✅ cas principal | Le modèle doit déjà réussir *parfois* : une récompense toujours nulle ne donne aucun signal |
| LLM de base | Llama-3.1-8B, Qwen2.5-7B, Mistral-7B (versions de base) | ⚠️ | Possible (R1-Zero) mais demande un modèle ≥ 7B et un format simple |
| Multimodal | Qwen2.5-VL | ✅ | `GRPOTrainer` accepte les images. RLVR sur des questions visuelles vérifiables |
| Agents / usage d'outils | Qwen2.5-Instruct, Llama-3.1-Instruct, Hermes 3 | ✅ | `GRPOTrainer(tools=..., environment_factory=...)` : récompense sur la réussite de la tâche multi-tours |
| Diffusion | Stable Diffusion, SDXL, Flux | ✅ | DDPO, Flow-GRPO : récompense esthétique ou d'alignement texte-image (bibliothèques dédiées) |
| Encodeur / classifieur | BERT, RoBERTa, ModernBERT | ❌ | Pas de génération |
| Modèle fermé via API | GPT-4o, Claude, Gemini | ⚠️ | OpenAI « Reinforcement Fine-Tuning » : on fournit des prompts et un grader |

## Quand l'utiliser

- ✅ Tâche **vérifiable** (maths, code, SQL, JSON valide, jeu) : RLVR, c'est le cas où le RL apporte le plus.
- ✅ Après SFT et/ou DPO, pour dépasser le niveau des démonstrations.
- ⚠️ Récompense apprise ou juge : risque de **reward hacking** (le modèle exploite les failles du juge). Garder la KL, surveiller les sorties.
- ❌ Si le modèle ne réussit jamais la tâche (récompense toujours 0) : d'abord SFT, ou une tâche plus facile (curriculum).

## Composants de l'exemple

```mermaid
flowchart TD
  classDef externe fill:#fed7aa,stroke:#ea580c,color:#7c2d12
  classDef code fill:#ffffff,stroke:#6b7280,color:#1f2937
  classDef gpu fill:#fecaca,stroke:#dc2626,color:#7f1d1d
  classDef disque fill:#dcfce7,stroke:#16a34a,color:#14532d

  Hub["Hugging Face Hub<br/>Qwen2.5-0.5B-Instruct, GSM8K"]
  Cache[("cache disque<br/>~/.cache/huggingface")]
  subgraph Py["Processus Python"]
    Run["run.py<br/>charge les questions<br/>et le modèle"]
    TRL["TRL GRPOTrainer<br/>algorithme DAPO"]
    Rec["recompenses.py<br/>exactitude, format"]
  end
  API["juge en option<br/>serveur d'API"]
  GPU["GPU CUDA, 12 Go"]
  Out[("sorties/05-rl-en-ligne/")]

  Hub -- "① HTTPS GET, 1re fois<br/>poids (1 Go), GSM8K" --> Cache
  Cache -- "② lecture des poids" --> Run
  Run -- "③ copie des poids<br/>④ ⑤ ⑩ ⑪ mesures" --> GPU
  Run -- "⑥ appel Python<br/>trainer.train()" --> TRL
  TRL -- "⑧ appel Python<br/>notation" --> Rec
  Rec -. "⑧ HTTP POST<br/>/v1/chat/completions" .-> API
  TRL -- "⑦ ⑨ noyaux CUDA<br/>génération, entraînement" --> GPU
  Run -- "⑫ ⑬ écriture" --> Out

  class Hub,API externe
  class Run,TRL,Rec code
  class GPU gpu
  class Cache,Out disque
```

`make run EX=SIGN05` vérifie les prérequis (modules, GPU), puis lance `run.py` dans l'environnement `.venv`. Le script lit `config.yaml` (modèle, algorithme `dapo`, source de récompense `regles`, génération, données, réglages) et charge GSM8K : 800 problèmes de mathématiques rédigés pour l'entraînement, et les 1 319 problèmes de test. Seul le résultat attendu sert pendant le RL : le modèle n'est jamais entraîné sur une solution écrite.

- **① Téléchargement.** Au premier lancement, les poids de Qwen2.5-0.5B-Instruct (environ 1 Go) et le jeu `openai/gsm8k` sont téléchargés depuis le Hugging Face Hub, par HTTPS.
- **② Lecture.** Tout est rangé dans le cache `~/.cache/huggingface` ; les lancements suivants le relisent sur le disque, sans réseau.
- **③ Copie sur le GPU.** Les poids sont copiés en fp32 dans la mémoire du GPU (`dtype: float32`), où ils restent jusqu'à la fin.
- **④ Mesure avant.** `run.py` appelle `commun/evaluation.py`, qui pose les 1 319 problèmes de test, puis 200 problèmes d'arithmétique de non-régression, par lots (batches) de 32.
- **⑤ Génération sur le GPU.** Pour chaque lot, le modèle raisonne token par token, jusqu'à 512 tokens, en décodage glouton. L'évaluation extrait ensuite le nombre final de chaque réponse et le compare au résultat attendu : c'est l'exactitude de départ.
- **⑥ Entraînement.** `run.py` confie le modèle, les prompts et les fonctions de récompense au GRPOTrainer de TRL, réglé pour DAPO (perte (loss) `dapo`, `epsilon_high` = 0,28, `beta` = 0, réponses tronquées ignorées). Rien n'est gelé : tous les poids du modèle sont entraînés.
- **⑦ Génération sur le GPU.** À chaque pas, le modèle lui-même produit 8 réponses par problème, par échantillonnage à température 1,0, jusqu'à 512 tokens chacune (environ 300 en moyenne).
- **⑧ Notation.** TRL passe les réponses à `recompenses.py` : 1 si le nombre final est juste, 0 sinon (`bonus_format: true` ajoute 0,2 quand la réponse se termine par `Answer: N`, utile sur l'arithmétique). Ce calcul se fait en Python, sans GPU. Avec `SET="recompense.source=juge recompense.api_base=http://localhost:11434/v1"`, la note vient d'un LLM juge interrogé par HTTP sur une API compatible OpenAI (flèche pointillée).
- **⑨ Forward et backward sur le GPU.** Chaque réponse reçoit un avantage : l'écart de sa récompense à la moyenne de son groupe de 8, divisé par l'écart type du groupe. Chaque pas traite 64 réponses, soit 8 problèmes (lots de 2, 32 accumulations) : passe avant (forward pass) pour les log-probabilités des tokens générés, perte DAPO, passe arrière (backward pass), mise à jour de tous les poids par AdamW 8 bits. 100 pas, taux d'apprentissage (learning rate) 1e-6, gradient checkpointing actif.
- **⑩ Mesure après.** Les mêmes problèmes sont posés au modèle entraîné.
- **⑪ Génération sur le GPU**, avec le même décodage qu'en ⑤.
- **⑫ Écriture du modèle** complet dans `sorties/05-rl-en-ligne/modele-dapo/`.
- **⑬ Écriture du rapport** dans `sorties/05-rl-en-ligne/resultats.json`, avec le même tableau affiché dans le terminal.

**Ce que fait le GPU.** Le processus Python ne calcule presque rien lui-même : PyTorch envoie au GPU des « noyaux CUDA », de petits programmes qui font les multiplications de matrices et l'attention sur les poids et les activations déjà présents dans sa mémoire. En génération (⑤, ⑦, ⑪), il y a une passe avant par token produit, accélérée par un cache des clés et des valeurs déjà calculées ; pendant l'entraînement, cette génération occupe l'essentiel du temps. En entraînement (⑨), chaque pas fait une passe avant, une passe arrière qui calcule les gradients de tous les poids, puis leur mise à jour par l'optimiseur AdamW 8 bits, dont l'état tient en 1 Go au lieu de 4. Avec `beta` = 0, aucun modèle de référence n'est chargé, et la récompense par règles ne demande aucun modèle : un seul modèle occupe la mémoire GPU. Les log-probabilités des réponses de 512 tokens sur un vocabulaire de 152 000 tokens imposent des lots de 2 réponses pour tenir en 12 Go.

**Ce qui est écrit.**

| Fichier | Contenu |
|---|---|
| `modele-dapo/model.safetensors` | Tous les poids du modèle entraîné (2,0 Go, en fp32) |
| `modele-dapo/config.json`, `generation_config.json` | Architecture et réglages de génération, pour recharger le modèle avec `AutoModelForCausalLM.from_pretrained` |
| `modele-dapo/tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja` | Le tokenizer et le format de conversation, copiés pour que le dossier se suffise à lui-même |
| `modele-dapo/training_args.bin` | Les réglages d'entraînement utilisés |
| `resultats.json` | Exactitude avant et après sur GSM8K et sur la non-régression, avec intervalle à 95 % et réponse de chaque question ; comparaisons appariées ; récompense moyenne en début et en fin d'entraînement ; algorithme et source de récompense |
| `config_effective.yaml` | La configuration réellement utilisée, surcharges `SET` comprises |

Le dossier de sortie prend le nom de l'algorithme : avec `SET="algo=grpo"`, il s'appelle `modele-grpo/`.

## Utilisation

```bash
python run.py --quick
python run.py                                                           # DAPO + récompense vérifiable, Qwen2.5-0.5B, GSM8K, ~1 h
python run.py --set algo=gspo
python run.py --set algo=rloo
python variantes/modele-recompense/run.py                               # 1) entraîner un modèle de récompense
python run.py --set recompense.source=modele \
  --set recompense.modele=sorties/05-rl-en-ligne--variantes--modele-recompense/modele   # 2) RLHF avec ce modèle
python run.py --set recompense.source=juge --set recompense.juge=Qwen/Qwen2.5-1.5B-Instruct       # RLAIF, juge local
python run.py --set recompense.source=juge --set recompense.api_base=http://localhost:11434/v1 --set recompense.juge=qwen3:8b   # juge servi par Ollama
python run.py --set peft.activer=true --set entrainement.lr=1e-5 --set modele.nom=Qwen/Qwen2.5-1.5B-Instruct
```

## Ce que fait l'exemple

**Le besoin** : faire mieux raisonner un modèle sur des problèmes dont on sait vérifier la réponse (calcul, code testable, extraction contrôlable), sans solutions rédigées : seul le résultat attendu est connu.

**Résultats** (Qwen2.5-0.5B-Instruct, RTX 5070 Ti Laptop ; 1 319 problèmes de test de GSM8K, 200 problèmes de non-régression ; 100 pas, 1 h d'entraînement) :

| | GSM8K | Non-régression (arithmétique) |
|---|---|---|
| Avant | 0,411 [0,385 ; 0,438] | 0,955 [0,917 ; 0,976] |
| **Après RL (DAPO)** | **0,465** [0,438 ; 0,492] | 0,955 [0,917 ; 0,976] |

Écart apparié sur les mêmes problèmes : +0,054 [+0,026 ; +0,082], p de McNemar = 0,0002. Non-régression : 0,000 [−0,030 ; +0,030]. Récompense moyenne pendant l'entraînement : 0,37 au début, 0,54 à la fin.

**Ce qu'on en retient.**

- **Le RL avec une récompense vérifiable fait progresser le modèle** de 5 points, sans aucune solution rédigée et sans rien perdre ailleurs.
- **Il fait mieux que les deux autres façons d'apprendre de la même tâche** mesurées dans ce dépôt, sur le même modèle et les mêmes problèmes de test : la distillation d'un modèle 6 fois plus gros ([SIGN06](../06-distillation) : 0,451) et l'entraînement sur ses propres solutions justes ([SIGN07](../07-donnees-synthetiques) : 0,387). Le RL apprend aussi de ses échecs, que le rejection sampling écarte.
- **Il reste modeste à ce budget** : 800 problèmes vus une fois. Les papiers (DeepSeekMath, DAPO) entraînent des milliers de pas, avec vLLM pour la génération.

## Variantes

### Variante modèle de récompense : apprendre à noter les réponses

**Le besoin** : quand la qualité d'une réponse ne se vérifie pas par un programme (utilité, ton, respect d'une consigne), le RL a besoin d'un modèle qui la note. C'est la première étape du RLHF : on l'entraîne sur des paires « réponse préférée (chosen) / réponse rejetée (rejected) ».

[`variantes/modele-recompense`](variantes/modele-recompense) ajoute une tête (head) scalaire à Qwen2.5-0.5B-Instruct et l'entraîne avec la perte de Bradley-Terry : la note de la réponse préférée doit dépasser celle de la rejetée. Les 4 000 paires d'entraînement viennent d'UltraFeedback (`trl-lib/ultrafeedback_binarized`, préférences d'un juge sur des réponses de modèles), une époque, lots de 4 × 4 avec gradient checkpointing. La mesure porte sur 500 paires du split `test` jamais vues : part des paires où le modèle note mieux la réponse préférée. Repère : le modèle de récompense publié Skywork-Reward-V2-Qwen3-0.6B, sur les mêmes paires.

| | Paires bien classées (500, 3 graines) |
|---|---|
| Tête aléatoire | 0,470 ± 0,031 |
| Après entraînement | **0,631 ± 0,017** |
| Modèle publié (Skywork 0,6B) | 0,704 [0,663 ; 0,742] |

Gain apparié sur les mêmes paires : +0,161 (de +0,120 à +0,202 selon la graine), p de McNemar < 10⁻³ ; l'intervalle de Wilson à 95 % de chaque graine exclut 0,5 (par exemple [0,593 ; 0,677]). Le modèle entraîné ici reste 0,073 sous le modèle publié (de −0,060 à −0,092 selon la graine, net) : un 0,5B entraîné sur 4 000 paires n'atteint pas le modèle publié.

Le modèle obtenu se branche comme récompense de l'exemple principal (`--set recompense.source=modele --set recompense.modele=…`, voir « Utilisation »), sert au best-of-N ([EXEC20](../20-calcul-inference)) ou classe des paires on-policy ([DPO itératif](../04-optimisation-preferences)). `donnees.source=hf` l'entraîne sur un jeu de préférences humaines (UltraFeedback, Skywork-Reward-Preference).

Réserve : un modèle de récompense se fait exploiter par la politique (réponses longues, flatterie) : il faut le surveiller pendant le RL. Le jeu synthétique d'arithmétique (`donnees.source=arith`, bonne réponse contre résultat décalé) reste disponible : une tâche facile, où le modèle apprend à vérifier une addition.

## Paramètres principaux (`config.yaml`)

| Clé | Rôle |
|---|---|
| `algo`, `beta` | Variante et pénalité KL (0 = pas de modèle de référence en mémoire) |
| `recompense.*` | Source de la récompense |
| `generation.num_generations` | Taille du groupe (4 à 16 en local, 16 à 64 à grande échelle) |
| `generation.max_completion_length` | Longueur de raisonnement autorisée |
| `generation.vllm` | Génération rapide (indispensable au-delà de quelques centaines de pas) |
| `recompense.bonus_format` | Bonus de format `Answer: N` (tâche d'arithmétique) |
| `entrainement.batch`, `accumulation` | Lot × accumulation multiple de `num_generations` |
| `evaluation.non_regression` | Tâche hors sujet mesurée avant et après |

## Grosse infra

- En local : faisable jusqu'à ~1,5B, mais la génération avec transformers est lente. C'est vLLM qui rend le RL praticable.
- [`infra/config.yaml`](infra/config.yaml) : DAPO sur Qwen2.5-7B, 16 réponses par prompt jusqu'à 4k tokens, vLLM colocalisé, ZeRO-3, 1 nœud 8×H100.
- [`infra/lancer.sh`](infra/lancer.sh) documente trois montages :
  - A : TRL + vLLM colocalisé ;
  - B : TRL + serveur vLLM dédié (`trl vllm-serve`) pour les modèles de 32B et plus ;
  - C : **verl** ou **OpenRLHF** (Ray, multi-nœuds, PPO avec critique, 70B+). Exemple de commande verl fourni.
- Ordres de grandeur : RLVR 7B sur maths, ~1 000 pas en 1 à 3 jours sur 8×H100. DeepSeek-R1 et Qwen3 : des milliers de GPU.

## Pièges

- **Récompense nulle partout** : aucun gradient. Vérifier `reward` dès les premiers pas, simplifier la tâche.
- **Reward hacking** : réponses vides mais bien formatées, répétition de ce qui plaît au juge… Lire des générations régulièrement.
- **Longueur** : la réponse peut s'allonger ou raccourcir sans raison. `dr_grpo` et `dapo` corrigent les biais connus.
- **Effondrement de l'entropie** : toutes les réponses d'un groupe deviennent identiques, avantage nul. Température ≥ 1, `epsilon_high` (DAPO).
- **Mémoire** : politique + référence (si β > 0) + génération. β=0 et LoRA réduisent l'empreinte.

## Références

- Ouyang et al., *InstructGPT* (2022) : RLHF avec PPO · Bai et al., *Constitutional AI* (2022) : RLAIF
- Shao et al., *DeepSeekMath / GRPO* (2024) · DeepSeek-AI, *DeepSeek-R1* (2025) · Lambert et al., *Tülu 3 : RLVR* (2024)
- Liu et al., *Dr. GRPO* (2025) · Yu et al., *DAPO* (2025) · Zheng et al., *GSPO* (2025) · Ahmadian et al., *Back to Basics: RLOO* (2024)
- verl : https://github.com/volcengine/verl · OpenRLHF : https://github.com/OpenRLHF/OpenRLHF
