# 07 — Données synthétiques et auto-amélioration

> Faire produire les données d'entraînement par un modèle (lui-même ou un autre), les **filtrer**, puis entraîner dessus.

| Regroupe | Modifie | Local 12 Go | Test `--quick` |
|---|---|---|---|
| Rejection sampling fine-tuning (RFT), STaR / auto-apprentissage, Self-Instruct / Evol-Instruct, critique-révision (phase supervisée de Constitutional AI), best-of-N distillé | Tous les poids (SFT final) | ✅ jusqu'à ~1B | ✅ |

## Principe

Schéma : [docs/schemas-methodes.md](../docs/schemas-methodes.md#m07) (ce que la méthode ajoute ou modifie dans le transformer).

Toutes les stratégies suivent le même pipeline en trois temps. Seule l'étape de génération change (`--set strategie=...`) :

```
générer  ->  filtrer / noter  ->  SFT (dossier 02, ou LoRA 03)
```

| `strategie` | Génération | Filtre |
|---|---|---|
| `rejet` (RFT / STaR) | Le modèle lui-même produit k solutions par question | On garde les solutions **vérifiées justes** |
| `self_instruct` | Un modèle générateur invente de **nouvelles questions** à partir d'exemples, puis y répond | Réponse vérifiable, déduplication, format valide |
| `critique_revision` | Le modèle répond, se critique selon un **principe écrit**, puis réécrit sa réponse | On garde la révision (entraînement sur « réponse révisée ») |

La clé est **le filtre**. Un modèle qui s'entraîne sur ses propres sorties non filtrées stagne ou se dégrade (model collapse). Filtrer par un vérificateur, un juge ou un modèle de récompense ne garde que ses meilleures sorties : le modèle s'améliore ainsi « par lui-même ». C'est le même ressort que le RL en ligne ([SIGN05](../05-rl-en-ligne)), en version hors ligne, plus simple et plus stable.

Lien avec la distillation ([SIGN06](../06-distillation)) : si le générateur est un modèle plus fort que l'élève (student), `self_instruct` et `rejet` deviennent de la distillation de séquences.

## Modèles concernés

| Type de modèle | Exemples de modèles | Applicable | Remarques |
|---|---|---|---|
| LLM décodeur instruct | Llama 3, Qwen2.5, Mistral 7B, Gemma 2 | ✅ cas principal | Le modèle doit réussir la tâche au moins de temps en temps (`rejet`), ou savoir suivre des consignes (`critique_revision`) |
| LLM de base | Llama-3.1-8B, Qwen2.5-7B, Mistral-7B (versions de base) | ⚠️ | Utiliser un générateur instruct distinct (`generateur.nom`) |
| Encodeur / classifieur | BERT, RoBERTa, ModernBERT | ✅ | Un LLM génère des textes étiquetés pour entraîner un petit classifieur (variante courante de `self_instruct`, voir *Grosse infra*) |
| Vision / multimodal | Qwen2.5-VL, LLaVA, Idefics | ✅ | Légendes générées par un VLM et filtrées par CLIP-score (recette de nombreux jeux image-texte) |
| Diffusion | Stable Diffusion, SDXL, Flux | ✅ | Prompts réécrits par un LLM (légendes synthétiques détaillées, recette DALL-E 3) |
| Modèle fermé via API | GPT-4o, Claude, Gemini | ✅ comme générateur | Générer avec GPT/Claude/Gemini pour entraîner un modèle ouvert, en respectant leurs conditions d'utilisation |

## Quand l'utiliser

- ✅ Peu ou pas de données annotées, mais une **tâche vérifiable** ou un bon générateur disponible.
- ✅ Spécialiser un petit modèle bon marché à partir d'un gros (souvent la méthode la plus rentable en production).
- ✅ Aligner sur des principes écrits sans annotateurs (`critique_revision`).
- ❌ Sans filtre fiable : le risque de dégradation l'emporte.

## Composants de l'exemple

```mermaid
flowchart TD
  classDef externe fill:#fed7aa,stroke:#ea580c,color:#7c2d12
  classDef code fill:#ffffff,stroke:#6b7280,color:#1f2937
  classDef gpu fill:#fecaca,stroke:#dc2626,color:#7f1d1d
  classDef disque fill:#dcfce7,stroke:#16a34a,color:#14532d

  Hub["Hugging Face Hub<br/>Qwen2.5-0.5B, GSM8K"]
  Cache[("cache disque<br/>~/.cache/huggingface")]
  subgraph Py["Processus Python"]
    Run["run.py<br/>charge les questions<br/>filtre les solutions"]
    Ev["evaluation.py<br/>modeles.py<br/>génération"]
    TRL["TRL SFTTrainer<br/>tous les poids"]
  end
  GPU["GPU CUDA, 12 Go"]
  Out[("sorties/<br/>07-donnees-synthetiques/")]

  Hub -- "① HTTPS GET, 1re fois<br/>poids (1 Go), GSM8K" --> Cache
  Cache -- "② lecture des poids" --> Run
  Run -- "③ copie des poids<br/>en mémoire GPU" --> GPU
  Run -- "④ ⑥ ⑪ appel Python<br/>mesure, échantillonnage" --> Ev
  Ev -- "⑤ ⑦ ⑫ noyaux CUDA<br/>génération" --> GPU
  Run -- "⑧ ⑬ écriture<br/>jsonl, modèle, résultats" --> Out
  Run -- "⑨ appel Python<br/>trainer.train()" --> TRL
  TRL -- "⑩ noyaux CUDA<br/>forward, backward" --> GPU

  GPU ~~~ Out
  class Hub externe
  class Run,Ev,TRL code
  class GPU gpu
  class Cache,Out disque
```

`make run EX=SIGN07` vérifie les prérequis (modules, GPU), puis lance `run.py` dans l'environnement `.venv`. Le script lit `config.yaml` (modèle de base, amorce, stratégie `rejet`, 1 tour, réglages de génération et d'entraînement) et charge GSM8K : 3 000 problèmes d'entraînement, dont seule la réponse finale sert (pour vérifier), et les 1 319 problèmes de test. Comme dans STaR et RFT, le modèle de départ est un modèle de base, qui n'a pas appris ce raisonnement : 4 problèmes résolus du jeu d'entraînement, placés en tête de chaque prompt (amorce few-shot), lui montrent le format attendu. Les solutions rédigées ne sont pas fournies : c'est le modèle lui-même qui les écrit, et seules les justes sont gardées.

- **① Téléchargement.** Au premier lancement, les poids de Qwen2.5-0.5B (environ 1 Go) et le jeu `openai/gsm8k` sont téléchargés depuis le Hugging Face Hub, par HTTPS.
- **② Lecture.** Tout est rangé dans le cache `~/.cache/huggingface` ; les lancements suivants le relisent sur le disque, sans réseau.
- **③ Copie sur le GPU.** Les poids sont copiés en float32 dans la mémoire du GPU. Le même modèle sert de générateur et d'élève (`generateur.nom: null`). Sans chat template, les prompts ont la forme « Question: … Answer: ».
- **④ Mesure de départ (tour 0).** `run.py` appelle `commun/evaluation.py`, qui pose les 1 319 problèmes de test, précédés de l'amorce, puis 200 problèmes d'arithmétique de non-régression, par lots (batches) de 32.
- **⑤ Génération sur le GPU.** Le modèle raisonne jusqu'à 512 tokens, en décodage glouton. Chaque suite s'arrête à la fin de sa ligne « Answer: … » : un modèle de base enchaînerait sinon sur du texte sans rapport. Le nombre final est comparé au résultat attendu.
- **⑥ Échantillonnage.** `run.py` demande 8 solutions par question pour les 3 000 questions, avec l'amorce, par lots de 8 questions.
- **⑦ Génération sur le GPU**, cette fois par tirage aléatoire à la température 0,8 : les 8 solutions d'une même question diffèrent.
- **⑧ Filtrage, rationalisation et écriture.** Pour chaque question, `run.py` garde les solutions dont le nombre final est juste et dont la suite de calculs diffère (RFT), au plus 4 par question pour ne pas surpondérer les questions faciles (ReST-EM). Les questions ratées repartent au GPU avec la réponse attendue en indice ; le modèle en rédige 4 raisonnements, et un raisonnement juste qui ne cite pas l'indice est gardé, associé à la question seule (STaR). Le corpus est écrit dans `donnees-tour-1.jsonl`.
- **⑨ Entraînement.** `run.py` confie le modèle et le corpus au SFTTrainer de TRL : fine-tuning complet, perte (loss) calculée sur la réponse seulement.
- **⑩ Forward et backward sur le GPU.** Lots de 4, accumulés 4 fois (16 exemples par pas), 1 époque, taux d'apprentissage (learning rate) 1e-5.
- **⑪ Remesure**, sur les mêmes problèmes, avec le même décodage qu'en ⑤, sans l'amorce : le modèle entraîné a appris le format.
- **⑫ Témoin (baseline).** Le modèle d'origine est rechargé et entraîné de la même façon sur les mêmes questions, avec une solution écrite par les auteurs de GSM8K pour chacune, puis mesuré.
- **⑬ Écriture du modèle et du rapport** : `sorties/07-donnees-synthetiques/modele/`, puis `resultats.json`, avec le même tableau affiché dans le terminal.

Avec `tours: 2` ou plus, les étapes ⑥ à ⑪ se répètent sur de nouvelles questions, en échantillonnant avec le modèle du tour précédent : c'est l'auto-amélioration itérée (STaR, ReST-EM).

**Ce que fait le GPU.** Le processus Python ne calcule presque rien lui-même : PyTorch envoie au GPU des « noyaux CUDA », de petits programmes qui font les multiplications de matrices et l'attention sur les poids et les activations déjà présents dans sa mémoire. En génération (⑤, ⑦, ⑫), il y a une passe avant (forward pass) par token produit, accélérée par un cache des clés et des valeurs déjà calculées ; à l'échantillonnage (⑦), chaque lot de 32 questions donne 256 suites produites en parallèle ; c'est l'essentiel de la durée. En entraînement (⑩), chaque pas fait une passe avant, une passe arrière (backward pass) qui calcule les gradients de tous les poids, puis leur mise à jour par l'optimiseur AdamW, dont l'état couvre tous les poids ; le gradient checkpointing recalcule les activations pendant la passe arrière au lieu de les garder. Le filtrage (⑧) se fait sur CPU, en Python. 

**Ce qui est écrit.**

| Fichier | Contenu |
|---|---|
| `donnees-tour-1.jsonl` | Le corpus synthétique, une ligne `{"question", "reponse", "cible"}` par solution gardée. Réutilisable par 02 ou 03 |
| `modele/model.safetensors` | Tous les poids du modèle après le dernier tour (1,98 Go, en float32) |
| `modele/config.json`, `generation_config.json` | Architecture Qwen2 et réglages de génération, pour recharger le modèle avec `AutoModelForCausalLM.from_pretrained` |
| `modele/tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja` | Le tokenizer et le format de conversation, copiés pour que le dossier se suffise à lui-même |
| `tour-1/`, `temoin/` | Dossiers de travail du trainer, vides : aucun point de contrôle (checkpoint) intermédiaire n'est sauvegardé |
| `resultats.json` | Pour le tour 0, le tour 1 et le témoin : exactitude sur GSM8K et sur la non-régression, avec intervalle à 95 % et réponse de chaque question ; part des questions résolues à l'échantillonnage, solutions et rationalisations gardées ; comparaison appariée `synthetique_contre_temoin` |
| `config_effective.yaml` | La configuration réellement utilisée, surcharges `SET` comprises |

## Utilisation

```bash
python run.py --quick
python run.py                                                  # RFT + STaR sur GSM8K, Qwen2.5-0.5B de base s'auto-entraîne
python run.py --set modele.nom=Qwen/Qwen2.5-0.5B-Instruct --set modele.format_texte=false --set amorce.n=0   # depuis le modèle instruct
python run.py --set strategie=self_instruct --set donnees.source=arith --set generateur.nom=Qwen/Qwen2.5-1.5B-Instruct
python run.py --set strategie=critique_revision --set donnees.source=arith
python run.py --set tours=3                                    # itérer (STaR / ReST-EM)
```

Le jeu généré est sauvegardé dans `sorties/.../donnees-tour-N.jsonl`, réutilisable par [DATA02](../02-fine-tuning-supervise) ou [DATA03](../03-peft) (`donnees.source=hf`, `donnees.nom=json`, `donnees.fichiers=...`).

## Ce que fait l'exemple

- **`rejet`** (par défaut) : le modèle produit 8 solutions par question et garde toutes les justes aux calculs distincts (RFT) ; les questions ratées sont rationalisées à partir de la réponse attendue (STaR) ; une époque de SFT, puis remesure. Un témoin est entraîné pareil sur les solutions humaines. `generation.toutes_justes: false` garde une seule solution juste par question.

**Résultats depuis Qwen2.5-0.5B-Instruct** (RTX 5070 Ti Laptop ; 1 319 problèmes de test de GSM8K ; 8 solutions par question, toutes les justes gardées) :

| | GSM8K | Non-régression (arithmétique) |
|---|---|---|
| Avant | 0,411 [0,385 ; 0,438] | 0,955 |
| RFT + STaR : 9 775 solutions écrites par le modèle | 0,393 [0,367 ; 0,419] | 0,92 |
| Une solution juste par question, 2 époques | 0,387 [0,361 ; 0,414] | 0,92 |
| Témoin : solutions humaines des mêmes questions | 0,280 [0,256 ; 0,305] | 0,825 |

Écarts appariés : RFT + STaR − avant −0,018 [−0,045 ; +0,008], p = 0,19, non significatif ; RFT + STaR − témoin +0,113, p < 0,001.

**Non démontré.** Ses propres solutions valent nettement mieux que les solutions humaines, trop laconiques, mais ne font pas progresser le modèle, qu'on en garde une ou toutes par question. Le modèle de départ est un modèle instruct déjà entraîné à ce raisonnement : 78 % des questions sont résolues en 8 tirages, et réapprendre ses propres solutions justes ne déplace pas son décodage glouton. Le RL en ligne ([SIGN05](../05-rl-en-ligne)), qui pénalise aussi les solutions fausses, le fait progresser (0,411 → 0,465). STaR et RFT partent de modèles de base amorcés en few-shot ; c'est le cadre de la configuration par défaut.

**Résultats depuis Qwen2.5-0.5B de base** (configuration par défaut, une graine/seed) :

| | GSM8K | Non-régression (arithmétique) |
|---|---|---|
| Avant, avec l'amorce | 0,363 [0,338 ; 0,390] | 0,465 |
| RFT + STaR : 5 720 solutions (au plus 4 par question) + 485 rationalisations | 0,402 [0,376 ; 0,428] | 0,87 |
| Témoin : solutions humaines des mêmes questions | 0,332 [0,307 ; 0,358] | 0,87 |

Écarts appariés : après − avant +0,039 [+0,013 ; +0,064], p = 0,004 ; après − témoin +0,070, p < 0,001.

**Non démontré à ce stade.** Ces chiffres sont à remesurer : la génération continuait après la ligne « Answer: » (texte parasite dans 5 403 des 6 205 solutions gardées, et nombre final parfois pris dans ce texte à la mesure de départ). L'arrêt à la fin de cette ligne corrige les deux ; la remesure se fait sur 3 graines.

- **`self_instruct`** : le générateur écrit de nouveaux problèmes du même style. On ne garde que ceux dont l'expression se calcule, et la réponse vient du calcul lui-même, donc elle est garantie juste.
- **`critique_revision`** : un principe (« être concis et donner le résultat final sous la forme Answer: N ») guide la critique puis la réécriture.

## Paramètres principaux (`config.yaml`)

| Clé | Rôle |
|---|---|
| `strategie` | `rejet`, `self_instruct` ou `critique_revision` |
| `generateur.nom` | `null` = le modèle entraîné lui-même (auto-amélioration) ; sinon un modèle plus fort |
| `generation.k`, `temperature` | Diversité des candidats |
| `generation.toutes_justes` | RFT : toutes les solutions justes aux calculs distincts, sinon une seule par question |
| `generation.rationalisation`, `k_rationalisation`, `indice` | STaR : raisonnements rédigés à partir de la réponse attendue pour les questions ratées |
| `tours` | Nombre d'itérations générer / filtrer / entraîner |
| `principes` | Liste de principes pour `critique_revision` |

## Grosse infra

- La génération domine le coût : **vLLM** sur un nœud (script [`../06-distillation/infra/generer_vllm.py`](../06-distillation/infra/generer_vllm.py), rejection sampling inclus), puis SFT FSDP ([02 infra](../02-fine-tuning-supervise/infra)).
- [`infra/lancer.sh`](infra/lancer.sh) : boucle STaR / ReST-EM à grande échelle (génération vLLM, filtre, SFT, et on recommence), plus les outils de génération de données dédiés : [distilabel](https://github.com/argilla-io/distilabel), [DataDreamer](https://github.com/datadreamer-dev/DataDreamer), NVIDIA NeMo Curator / Nemotron-4-340B pour la génération et la notation.
- Ordres de grandeur : les jeux synthétiques récents (OpenMathInstruct-2, 14 M solutions ; Tülu 3 SFT mix) demandent des milliers d'heures GPU de génération.

## Pièges

- **Model collapse** : jamais sans filtre. Garder une part de données réelles.
- **Faux positifs du vérificateur** : bonne réponse obtenue par un raisonnement faux. Préférer plusieurs solutions justes et diverses.
- **Biais vers le facile** : on ne garde que les questions déjà résolues, souvent celles que le modèle réussissait sans aide. Il faut des solutions aux questions difficiles (plus d'échantillons, rationalisation STaR).
- **Diversité** : Self-Instruct converge vite vers les mêmes questions. Dédupliquer (n-grammes, embeddings), varier les exemples d'amorce.
- **Contamination** : vérifier que les données générées ne recopient pas le jeu de test.
- **Licences** : les conditions de certains fournisseurs interdisent d'entraîner un modèle concurrent sur leurs sorties.

## Références

- Zelikman et al., *STaR* (2022) · Yuan et al., *Scaling Relationship… RFT* (2023) · Singh et al., *ReST-EM* (2023)
- Wang et al., *Self-Instruct* (2022) · Xu et al., *WizardLM / Evol-Instruct* (2023)
- Bai et al., *Constitutional AI* (2022) : phase critique-révision · Shumailov et al., *Model collapse* (2024)
