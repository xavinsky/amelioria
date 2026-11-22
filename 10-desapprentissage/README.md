# 10 — Désapprentissage (machine unlearning)

> Faire **oublier** au modèle un ensemble précis de connaissances (données personnelles, contenu sous copyright, savoir dangereux), sans détruire le reste.

| Regroupe | Modifie | Local 12 Go | Test `--quick` |
|---|---|---|---|
| Gradient ascent, gradient difference, NPO (+ rétention), réponses « je ne sais pas » (IDK). En infra : SimNPO, RMU (open-unlearning) | Tous les poids | ✅ jusqu'à ~1B | ✅ |

## Principe

Schéma : [docs/schemas-methodes.md](../docs/schemas-methodes.md#m10) (ce que la méthode ajoute ou modifie dans le transformer).

On dispose d'un **ensemble à oublier** (forget set) et d'un **ensemble à retenir** (retain set). Toutes les méthodes suivent la même boucle : un lot à oublier, éventuellement un lot à retenir, puis une perte (loss) combinée. Seule la perte change (`--set desapprentissage.methode=...`) :

| Méthode | Perte sur l'oubli | Rétention | Comportement |
|---|---|---|---|
| `gradient_ascent` | `-NLL` : on *monte* la perte | non | Oublie vite, mais s'effondre vite (le modèle finit par tout désapprendre) |
| `gradient_difference` | `-NLL` | `+NLL` sur le retain set | Plus stable |
| `npo` | Perte de type DPO sans réponse préférée (chosen) : `-(2/β)·log σ(-β·log(π/π_ref))` | optionnelle (`retenu: true`) | Bornée : l'oubli ralentit tout seul, pas d'effondrement. Référence actuelle |
| `idk` | SFT vers « je n'ai pas d'information » | `+NLL` | Refus explicite plutôt qu'oubli des poids |

Le protocole suit **TOFU** : des personnages fictifs sont d'abord appris (*implantation*), puis on fait oublier 25 % d'entre eux (10 personnages sur 40). Comme le modèle ne pouvait pas les connaître avant, on sait exactement ce qui doit disparaître. Cette phase d'implantation est incluse dans `run.py`.

Mesures :

| Mesure | Rôle |
|---|---|
| `oubli` (exactitude) et `nll_oubli` | Doivent respectivement baisser et monter |
| `retenu` et `nll_retenu` | Doivent rester stables |
| `perplexite_generale` | Non-régression sur les connaissances générales |

## Modèles concernés

| Type de modèle | Exemples de modèles | Applicable | Remarques |
|---|---|---|---|
| LLM décodeur (toutes familles) | Llama 3, Qwen2.5, Mistral 7B, Gemma 2 | ✅ cas principal | [`run.py`](run.py) |
| LLM avec LoRA | Llama 3, Qwen2.5 + adaptateur LoRA | ✅ | Désapprentissage sur adaptateur : réversible, moins destructeur |
| Classifieurs (vision, texte) | BERT, ViT, ResNet | ✅ | Même idée (gradient ascent sur les exemples à oublier, SCRUB, fine-tuning sur le retain set) |
| Diffusion | Stable Diffusion, SDXL, Flux | ✅ très étudié | Oublier un style, un concept, une personne : ESD, concept ablation, UCE |
| Modèle fermé via API | GPT-4o, Claude, Gemini | ❌ | Seulement des filtres en entrée et en sortie |

## Quand l'utiliser

- ✅ Obligation de retrait d'une donnée (RGPD, droit à l'oubli), contenu sous licence retiré.
- ✅ Réduire des connaissances dangereuses (WMDP : RMU).
- ⚠️ Le désapprentissage approché n'offre **aucune garantie formelle** : l'information peut revenir par extraction ou par un léger fine-tuning. La seule garantie complète est de réentraîner sans la donnée.
- ⚠️ Si l'objectif est seulement de ne pas *dire* quelque chose : un filtre ou un refus (`idk`) peut suffire.

## Composants de l'exemple

```mermaid
flowchart TD
  classDef externe fill:#fed7aa,stroke:#ea580c,color:#7c2d12
  classDef code fill:#ffffff,stroke:#6b7280,color:#1f2937
  classDef gpu fill:#fecaca,stroke:#dc2626,color:#7f1d1d
  classDef disque fill:#dcfce7,stroke:#16a34a,color:#14532d

  Hub["Hugging Face Hub<br/>Qwen2.5-0.5B-Instruct<br/>wikitext-2"]
  Cache[("cache disque<br/>~/.cache/huggingface")]
  subgraph Py["Processus Python"]
    Run["run.py<br/>génère les personnages<br/>boucle NPO"]
    TRL["TRL SFTTrainer<br/>implantation"]
    Ev["mesurer()<br/>evaluation.py"]
  end
  GPU["GPU CUDA, 12 Go"]
  Out[("sorties/<br/>10-desapprentissage/")]

  Hub -- "① HTTPS GET, 1re fois<br/>poids (1 Go), textes" --> Cache
  Cache -- "② lecture des poids" --> Run
  Run -- "④ appel Python<br/>trainer.train()" --> TRL
  TRL -- "⑤ noyaux CUDA<br/>forward, backward" --> GPU
  Run -- "⑥ ⑨ appel Python<br/>mesure avant, après" --> Ev
  Ev -- "⑦ ⑩ noyaux CUDA<br/>génération, forward" --> GPU
  Run -- "③ copie des poids<br/>⑧ NPO, forward, backward" --> GPU
  Run -- "⑪ ⑫ écriture" --> Out
  GPU ~~~ Out

  class Hub externe
  class Run,TRL,Ev code
  class GPU gpu
  class Cache,Out disque
```

`make run EX=EDIT10` vérifie les prérequis (modules, GPU), puis lance `run.py` dans l'environnement `.venv`. Le script lit `config.yaml` (modèle, données, implantation, désapprentissage, évaluation) et génère en mémoire, sans téléchargement, 40 personnages fictifs aux noms inventés, avec 4 questions chacun (ville, année de naissance, métier, création) : 160 questions. 25 % des personnages, soit 10 personnages et 40 questions, sont tirés pour être oubliés ; 25 % autres sont tenus à l'écart (jamais vus par la perte, mesure de rétention hors perte) ; les 20 restants servent à la perte de rétention. L'oubli est aussi mesuré sur des reformulations et en français, et l'entraînement s'arrête dès que l'exactitude sur l'oubli passe sous 5 %.

- **① Téléchargement.** Au premier lancement, transformers télécharge les poids de Qwen2.5-0.5B-Instruct (environ 1 Go, safetensors) depuis le Hugging Face Hub, par HTTPS. À la première mesure, la bibliothèque datasets télécharge aussi wikitext-2, qui sert à la perplexité générale.
- **② Lecture.** Poids et textes sont rangés dans le cache `~/.cache/huggingface` ; les lancements suivants les relisent sur le disque, sans réseau.
- **③ Copie sur le GPU.** Les poids sont copiés en fp32 dans la mémoire du GPU, où ils restent jusqu'à la fin.
- **④ Implantation.** Le modèle ne connaît pas les personnages inventés : `run.py` les lui apprend d'abord avec le SFTTrainer de TRL, sur les 160 questions, pendant 8 époques (lots (batches) de 8, accumulation 2, taux d'apprentissage (learning rate) 2e-5). Tous les poids sont entraînés.
- **⑤ Forward et backward sur le GPU.** La perte porte sur la réponse seulement, pas sur la question. Le gradient checkpointing recalcule les activations pendant la passe arrière (backward pass) au lieu de les garder.
- **⑥ Mesure avant.** `run.py` mesure trois choses : la vraisemblance des bonnes réponses (`nll_oubli`, `nll_retenu`), l'exactitude en génération sur les 16 questions à oublier et sur 80 questions retenues, et la perplexité sur 40 paragraphes du split `test` de wikitext-2.
- **⑦ Génération et passes avant sur le GPU.** Chaque réponse est générée jusqu'à 24 tokens, par lots de 16 ; elle est juste si elle contient la réponse attendue. La vraisemblance et la perplexité ne demandent qu'une passe avant (forward pass) par lot.
- **⑧ Désapprentissage NPO.** La boucle est écrite dans `run.py`, sans Trainer : 60 pas, lots de 8, AdamW au taux 1e-5, gradients bornés à 1. À chaque pas, un lot de questions à oublier reçoit la perte NPO (β = 0,1), qui pousse la probabilité des bonnes réponses sous celle d'une copie figée du modèle, prise juste après l'implantation. Un lot de questions retenues ajoute la perte habituelle, de poids 1, pour garder le reste.
- **⑨ Mesure après.** Les mêmes questions et les mêmes paragraphes sont mesurés sur le modèle modifié.
- **⑩ Génération et passes avant sur le GPU**, avec le même décodage qu'en ⑦.
- **⑪ Écriture du modèle** dans `sorties/10-desapprentissage/modele-npo/`.
- **⑫ Écriture du rapport** dans `sorties/10-desapprentissage/resultats.json`, avec le même tableau affiché dans le terminal.

**Ce que fait le GPU.** Le processus Python ne calcule presque rien lui-même : PyTorch envoie au GPU des « noyaux CUDA », de petits programmes qui font les multiplications de matrices et l'attention sur les poids et les activations déjà présents dans sa mémoire. En génération (⑦, ⑩), il y a une passe avant par token produit, accélérée par un cache des clés et des valeurs déjà calculées. En entraînement (⑤, ⑧), chaque pas fait une passe avant, une passe arrière qui calcule les gradients de tous les poids, puis la mise à jour par AdamW, dont l'état couvre tout le modèle. Les poids maîtres restent en fp32, les calculs se font en bf16. Pendant ⑧, le GPU porte aussi la copie de référence de NPO, en bf16 et sans gradients : pour chaque lot à oublier, elle fait une passe avant de plus.

**Ce qui est écrit.**

| Fichier | Contenu |
|---|---|
| `modele-npo/model.safetensors` | Le modèle complet après implantation puis désapprentissage (1,9 Go, en fp32) |
| `modele-npo/config.json`, `generation_config.json` | L'architecture et les réglages de génération, pour recharger le modèle avec `AutoModelForCausalLM.from_pretrained` |
| `modele-npo/tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja` | Le tokenizer et le format de conversation, copiés pour que le dossier se suffise à lui-même |
| `implantation/` | Dossier de travail du SFTTrainer, vide : aucun point de contrôle (checkpoint) n'est sauvegardé |
| `resultats.json` | Pour `avant` et `apres` : `nll_oubli`, `nll_retenu`, `perplexite_generale`, et l'exactitude `oubli` et `retenu` avec trois réponses d'exemple ; la liste `entites_oubliees` des 4 personnages oubliés |
| `config_effective.yaml` | La configuration réellement utilisée, surcharges `SET` comprises |

Avec `SET="sauvegarder=false"`, le dossier `modele-npo/` n'est pas écrit. Avec une autre méthode (`desapprentissage.methode`), il prend son nom : `modele-gradient_ascent/`, `modele-idk/`...

## Utilisation

```bash
python run.py --quick
python run.py                                                          # idk + rétention, Qwen2.5-0.5B, 80 personnages dont 10 oubliés
python run.py --set desapprentissage.methode=gradient_ascent           # observer l'effondrement de `retenu`
python run.py --set desapprentissage.methode=gradient_difference
python run.py --set desapprentissage.methode=idk
python run.py --set donnees.part_oubli=0.05 --set entrainement.max_steps=120
python run.py --set donnees.source=tofu --set implantation.activer=false --set modele.nom=open-unlearning/tofu_Llama-3.2-1B-Instruct_full
```

## Ce que fait l'exemple

1. Génère 80 personnages (320 questions) et les répartit en trois groupes : 10 à oublier (40 questions), 20 tenus à l'écart (jamais vus par la perte de désapprentissage), 50 pour la perte de rétention.
2. **Implantation** : SFT sur toutes les questions. Le modèle connaît alors les 40 personnages.
3. Mesure l'exactitude sur les personnages à oublier (question d'origine, reformulée, en français), sur les personnages tenus à l'écart (rétention hors perte, question d'origine et reformulée), sur ceux de la perte de rétention, la part de sorties dégénérées et la perplexité générale.
4. `idk` + rétention (taux 2e-6, poids de rétention 5), 120 pas au plus, arrêt dès que l'exactitude sur l'oubli passe sous 5 % (contrôle tous les 2 pas). `--set desapprentissage.methode=npo` (ou `gradient_difference`, `gradient_ascent`) change la méthode.
5. Remesure, 3 graines (`make graines EX=10 N=3`).

Les tableaux ci-dessous donnent, dans l'ordre, la mesure de NPO + rétention sur 40 personnages, la comparaison de 12 réglages, la mesure finale d'`idk` sur 40 personnages, puis celle d'`idk` sur 80 personnages (configuration par défaut).

**Résultats de NPO + rétention** (40 personnages dont 10 à oublier, 20 dans la perte de rétention ; Qwen2.5-0.5B-Instruct, 3 graines, moyenne) :

| | Avant | Après NPO + rétention |
|---|---|---|
| Exactitude sur les personnages oubliés | 0,992 | **0,017** |
| ... reformulée | 0,767 | 0,017 |
| ... en français | 0,933 | 0,008 |
| Personnages tenus à l'écart (rétention hors perte) | 1,000 | **0,067** |
| ... reformulée | 0,758 | 0,058 |
| Personnages de la perte de rétention | 0,996 | 0,221 |
| Part de sorties dégénérées (oubli) | 0 | 0,25 |
| Perplexité générale | 25,8 | 27,0 |

Les écarts avant/après sont nets sur chaque graine pour toutes les lignes d'exactitude, dans les deux sens : l'oubli est obtenu, mais la rétention s'effondre aussi, y compris sur les personnages utilisés dans la perte de rétention (0,996 → 0,221). La perplexité générale ne bouge presque pas (25,8 → 27,0) : le modèle reste un modèle de langue, mais il a perdu les faits de **tous** les personnages appris. Le désapprentissage n'est pas sélectif dans cette configuration : la variante reste « non prouvée ».

**Comparaison des méthodes et des réglages** (12 combinaisons sur les graines de réglage 42 à 44 : méthodes `npo`, `idk` et `gradient_difference`, taux d'apprentissage 2e-6 et 5e-6, poids de rétention 1 et 5, contrôle de l'oubli tous les 2 pas) : NPO et `gradient_difference` laissent la rétention hors perte entre 0,25 et 0,58 dans tous les réglages (même quand les personnages de la perte de rétention sont conservés : la perte protège ce qu'elle voit, pas ce qu'elle ne voit pas). `idk` (apprendre à répondre « I'm not sure, I don't have information about this person. » aux questions sur les personnages à oublier) la conserve entre 0,90 et 0,975, sans sortie dégénérée. Le réglage retenu par une règle fixée avant la mesure (oubli sous 5 %, meilleure rétention hors perte) est `idk`, taux 2e-6, poids de rétention 5.

**Mesure finale de ce réglage, sur trois graines neuves** (45, 46, 47 ; 40 questions à oublier et 40 questions tenues à l'écart par graine) :

| | Avant | Après `idk` |
|---|---|---|
| Personnages à oublier | 1,000 | **0,042** (0,050 ; 0,025 ; 0,050) |
| ... reformulée | 0,750 | **0,225** (écart de −0,48 à −0,60, net sur chaque graine) |
| ... en français | 0,883 | **0,183** |
| Personnages tenus à l'écart | 0,992 | **0,825** (0,950 ; 0,600 ; 0,925) |
| ... reformulée | 0,750 | 0,617 |
| Personnages de la perte de rétention | 0,992 | 0,942 |
| Sorties dégénérées | 0 | 0 |
| Perplexité générale | 26,0 | 25,8 |

L'oubli est net sur chaque graine, y compris sur les reformulations et en français, sans sortie dégénérée ni dégradation de la perplexité. La rétention tient sur deux graines (écart de −0,03 et −0,08, non significatifs) mais pas sur la graine 46 (0,992 → 0,600, écart −0,40 [−0,55 ; −0,25]). Le désapprentissage reste « non prouvé » : il n'est pas sélectif de façon reproductible. Les faits oubliés restent en partie retrouvables par reformulation (0,225 contre 0,75 au départ), et `idk` apprend un refus plutôt qu'il n'efface l'information.

**Avec plus de personnages dans le jeu de rétention** (80 personnages : 10 à oublier, 20 tenus à l'écart, 50 dans la perte de rétention ; `idk`, taux 2e-6, poids de rétention 5 ; trois graines neuves 48, 49, 50 ; 40 questions à oublier et 80 questions tenues à l'écart par graine) :

| | Avant | Après `idk` |
|---|---|---|
| Personnages à oublier | 0,983 | **0,025** (0,025 ; 0,000 ; 0,050) |
| ... reformulée | 0,942 | **0,242** (écart de −0,65 à −0,78, net sur chaque graine) |
| Personnages tenus à l'écart | 0,996 | **0,954** (0,975 ; 0,963 ; 0,925) |
| ... reformulée | 0,946 | 0,867 |
| Personnages de la perte de rétention | 0,981 | 0,978 |
| Sorties dégénérées | 0 | 0 |
| Perplexité générale | 29,1 | 29,0 |

Sur chaque graine, l'oubli est net sur la question d'origine et sur la reformulation, la rétention hors perte reste à au moins 92 % du départ (écart de −0,03, −0,03 et −0,08), aucune sortie n'est dégénérée et la perplexité ne bouge pas. Avec 20 personnages dans la perte de rétention (mesure sur 40 personnages, plus haut), la rétention hors perte tombe à 0,600 sur une graine sur trois ; avec 50 personnages, elle reste à au moins 0,925 sur chaque graine : un jeu de rétention plus large la fait généraliser aux personnages non vus par la perte. Réserves : `idk` apprend un refus, il n'efface pas l'information (24 % des questions reformulées sur les faits oubliés obtiennent encore la bonne réponse) ; l'échelle est celle de personnages inventés appris par un modèle de 0,5B.

## Paramètres principaux (`config.yaml`)

| Clé | Rôle |
|---|---|
| `desapprentissage.methode` | Voir le tableau |
| `desapprentissage.beta` | NPO : plus grand = arrêt plus précoce |
| `desapprentissage.poids_retenu` | Équilibre entre oubli et rétention |
| `entrainement.max_steps`, `lr` | Trop de pas, et tout s'effondre (surtout en gradient ascent) |
| `donnees.source` | `faits` (généré) ou `tofu` (benchmark réel) |

## Grosse infra

- [`infra/config.yaml`](infra/config.yaml) : TOFU forget10 sur le modèle Llama-3.1-8B déjà fine-tuné sur TOFU (1 GPU 80 Go).
- [`infra/lancer.sh`](infra/lancer.sh) : [open-unlearning](https://github.com/locuslab/open-unlearning), le cadre de référence. Benchmarks TOFU, MUSE, WMDP ; méthodes GradAscent, GradDiff, NPO, SimNPO, DPO, RMU ; multi-GPU via accelerate/DeepSpeed ; métriques standard (forget quality, model utility, attaques par extraction).

## Pièges

- **Effondrement** : le gradient ascent sans rétention finit par produire du charabia. Toujours surveiller `retenu` et la perplexité générale.
- **Oubli superficiel** : la probabilité baisse sur la formulation exacte, mais une paraphrase ou quelques pas de fine-tuning font revenir l'information. Évaluer avec des paraphrases et des attaques.
- **Retain set mal choisi** : il doit contenir des connaissances *proches* de celles à oublier (même domaine), sinon on oublie aussi les voisins.

## Références

- Maini et al., *TOFU* (2024) · Zhang et al., *Negative Preference Optimization* (2024) · Fan et al., *SimNPO* (2024)
- Li et al., *WMDP / RMU* (2024) · Shi et al., *MUSE* (2024) · Eldan & Russinovich, *Who's Harry Potter?* (2023)
- Gandikota et al., *Erasing Concepts from Diffusion Models* (2023)
