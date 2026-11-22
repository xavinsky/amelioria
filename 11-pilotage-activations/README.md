# 11 — Pilotage par activations (steering vectors, abliteration)

> Repérer la direction d'un concept dans les activations internes du modèle, puis l'**ajouter** pendant la génération (sans toucher aux poids) ou la **retirer** des poids (sans entraînement).

| Regroupe | Modifie | Local 12 Go | Test `--quick` |
|---|---|---|---|
| Steering vectors / Activation Addition (ActAdd), Contrastive Activation Addition (CAA), representation engineering (RepE), control vectors, abliteration (orthogonalisation des poids) | Rien (mode vecteur) ou quelques matrices projetées (mode ablation) | ✅ jusqu'à ~7B en bf16 (inférence seule) | ✅ |

## Principe

Schéma : [docs/schemas-methodes.md](../docs/schemas-methodes.md#m11) (ce que la méthode ajoute ou modifie dans le transformer).

Dans un transformer, le **flux résiduel** accumule les représentations couche après couche. Beaucoup de concepts (sentiment, formalité, langue, vérité, refus…) y sont codés approximativement par une **direction linéaire**.

1. **Extraire la direction** : on passe des paires de textes contrastés (positif / négatif) et on prend, à une couche donnée, la différence des activations moyennes au dernier token : `d = moyenne(h⁺) - moyenne(h⁻)`.
2. Deux usages :
   - **`mode: vecteur`** : un crochet ajoute `α·d` à la sortie de la couche à chaque token généré. α > 0 pousse vers le concept, α < 0 vers son contraire. Réversible, réglable à la volée, sans aucune modification des poids.
   - **`mode: ablation`** : on projette la direction hors de toutes les matrices qui écrivent dans le flux résiduel (`W ← (I - d̂d̂ᵀ)W` pour `o_proj`, `down_proj` et les embeddings). Le modèle ne peut plus représenter ce concept. Le checkpoint obtenu est normal : il se sert avec vLLM, llama.cpp, etc.

L'exemple utilise le **sentiment**, mesuré par un classifieur externe (DistilBERT SST-2) sur deux types de consignes : neutres, et demandant un texte négatif.

## Modèles concernés

| Type de modèle | Exemples de modèles | Applicable | Remarques |
|---|---|---|---|
| LLM décodeur | LLaMA, Qwen, Mistral, Gemma, Phi, GPT-2 | ✅ cas principal | [`run.py`](run.py) trouve les couches automatiquement (`commun/modeles.py:couches_decodeur`) |
| MoE | Mixtral 8x7B, Qwen3-30B-A3B, DeepSeek-V3 | ✅ | Le flux résiduel est partagé : même méthode |
| Multimodal | LLaVA, Qwen-VL | ✅ | Pilotage (steering) de la partie langage |
| Encodeurs | BERT | ⚠️ | Directions extractibles (sondes linéaires), pilotage moins utile sans génération |
| Diffusion | Stable Diffusion, SDXL, Flux | ✅ | Directions dans l'espace des embeddings de texte ou dans l'UNet (« concept sliders ») |
| Modèle fermé via API | GPT-4o, Claude, Gemini | ❌ | Pas d'accès aux activations |

## Quand l'utiliser

- ✅ Ajuster un trait de style (ton, formalité, verbosité, langue) sans données d'entraînement ni GPU d'entraînement.
- ✅ Recherche et interprétabilité : vérifier qu'un concept est représenté linéairement, et où.
- ✅ Retirer de façon permanente un comportement indésirable très ciblé (ablation), quand l'entraînement est impossible.
- ⚠️ Un α trop grand dégrade la cohérence : il faut chercher le bon compromis.
- ❌ Pour des capacités nouvelles ou des connaissances : le pilotage réoriente ce qui existe déjà, il n'ajoute rien.

## Composants de l'exemple

```mermaid
flowchart TD
  classDef externe fill:#fed7aa,stroke:#ea580c,color:#7c2d12
  classDef code fill:#ffffff,stroke:#6b7280,color:#1f2937
  classDef gpu fill:#fecaca,stroke:#dc2626,color:#7f1d1d
  classDef disque fill:#dcfce7,stroke:#16a34a,color:#14532d

  Hub["Hugging Face Hub<br/>Qwen2.5-1.5B-Instruct<br/>DistilBERT SST-2"]
  Cache[("cache disque<br/>~/.cache/huggingface")]
  subgraph Py["Processus Python"]
    Run["run.py<br/>direction d, α<br/>génération"]
    Hook["forward hook<br/>couche 14 sur 28"]
    Clf["pipeline transformers<br/>classifieur"]
  end
  GPU["GPU CUDA, 12 Go"]
  Out[("sorties/<br/>11-pilotage-activations/")]

  Hub -- "① HTTPS GET, 1re fois<br/>poids (3 Go + 270 Mo)" --> Cache
  Cache -- "② lecture des poids" --> Run
  Run -- "④ ⑨ appel Python<br/>pose du hook" --> Hook
  Hook -- "⑤ ⑩ hook sur les activations<br/>lecture, puis ajout α·d" --> GPU
  Run -- "③ copie des poids<br/>⑥ génération" --> GPU
  Run -- "⑦ ⑪ appel Python<br/>textes à juger" --> Clf
  Clf -- "⑧ ⑫ noyaux CUDA<br/>classification" --> GPU
  Run -- "⑬ écriture" --> Out
  GPU ~~~ Out

  class Hub externe
  class Run,Hook,Clf code
  class GPU gpu
  class Cache,Out disque
```

`make run EX=EDIT11` vérifie les prérequis (modules, GPU), puis lance `run.py` dans l'environnement `.venv`. Le script lit `config.yaml` (modèle, mode `vecteur`, couche, intensités α, évaluation). Toutes les phrases sont écrites dans `run.py`, sans téléchargement : 15 paires de phrases de sentiment opposé (« I absolutely love the weather… » / « I absolutely hate the weather… »), 30 consignes neutres (« Write one sentence about the movie. ») et 30 demandes de texte négatif (« Write an angry complaint about the hotel. »).

- **① Téléchargement.** Au premier lancement, transformers télécharge depuis le Hugging Face Hub, par HTTPS, les poids de Qwen2.5-1.5B-Instruct (environ 3 Go) et ceux du classifieur de sentiment `distilbert-base-uncased-finetuned-sst-2-english` (environ 270 Mo).
- **② Lecture.** Les poids sont rangés dans le cache `~/.cache/huggingface` ; les lancements suivants les relisent sur le disque, sans réseau.
- **③ Copie sur le GPU.** Les deux modèles sont copiés dans la mémoire du GPU, le LLM en bf16. Ses poids sont gelés : rien n'est entraîné.
- **④ Pose du hook de lecture.** `run.py` accroche un *forward hook* PyTorch à la sortie du bloc 14 sur 28, au milieu du modèle (`couche_relative: 0.5`).
- **⑤ Extraction de la direction.** Chacune des 30 phrases passe dans le modèle ; à chaque passage, le hook copie l'activation du flux résiduel au dernier token. La direction `d` est la moyenne des activations positives moins celle des négatives. Le hook est ensuite retiré.
- **⑥ Génération sans intervention.** Le modèle répond aux 30 consignes neutres et aux 30 demandes (prompts) négatives, en génération gloutonne jusqu'à 48 tokens, par lots (batches) de 16.
- **⑦ Jugement.** Les textes produits sont passés au classifieur, via la `pipeline("sentiment-analysis")` de transformers.
- **⑧ Classification sur le GPU.** Le classifieur rend POSITIVE ou NEGATIVE pour chaque texte ; le résultat est le taux de textes positifs.
- **⑨ Pose du hook de pilotage**, une fois pour chaque α de la liste `[-2.0, 1.0, 2.0]`.
- **⑩ Génération pilotée.** Les mêmes 60 consignes sont reposées, avec le même décodage qu'en ⑥. À chaque passe avant (forward pass), le hook remplace la sortie du bloc 14 par cette sortie plus α·d, à toutes les positions. Les poids ne changent pas : retirer le hook rend le modèle d'origine.
- **⑪ Jugement** des textes pilotés.
- **⑫ Classification sur le GPU**, comme en ⑧.
- **⑬ Écriture** de la direction dans `sorties/11-pilotage-activations/direction.pt` et du rapport dans `resultats.json`, avec le même tableau affiché dans le terminal.

**Ce que fait le GPU.** Le processus Python ne calcule presque rien lui-même : PyTorch envoie au GPU des « noyaux CUDA », de petits programmes qui font les multiplications de matrices et l'attention sur les poids et les activations déjà présents dans sa mémoire. Il n'y a ni passe arrière (backward pass) ni optimiseur : tout le travail est de l'inférence. En génération (⑥, ⑩), il y a une passe avant par token produit, accélérée par un cache des clés et des valeurs déjà calculées. Le hook est une fonction Python appelée par PyTorch au milieu de chaque passe avant ; l'addition α·d qu'elle demande est elle-même un noyau CUDA sur le tenseur d'activations, qui ne quitte pas la mémoire du GPU. Le classifieur (⑧, ⑫) ne fait qu'une passe avant par lot de textes.

**Ce qui est écrit.**

| Fichier | Contenu |
|---|---|
| `direction.pt` | La direction `d`, un vecteur de la taille du flux résiduel (fichier de 8 Ko), le numéro de couche et le nom du modèle : de quoi piloter à nouveau ce modèle sans réextraire |
| `resultats.json` | Pour `sans_intervention` et pour chaque `alpha_-2.0`, `alpha_1.0`, `alpha_2.0` : le `taux_positif` sur les consignes neutres et sur les demandes négatives, avec un texte d'exemple de chaque côté |
| `config_effective.yaml` | La configuration réellement utilisée, surcharges `SET` comprises |

Avec `SET="pilotage.mode=ablation"`, la direction est retirée des matrices qui écrivent dans le flux résiduel, et le dossier `modele-ablation/` contient le modèle complet aux poids modifiés, à la place de `direction.pt`.

## Utilisation

```bash
python run.py --quick
python run.py                                                  # ablation de la direction de sentiment, Qwen2.5-1.5B-Instruct
python run.py --set pilotage.couche_relative=0.3               # la couche change beaucoup l'effet
python run.py --set pilotage.mode=vecteur                       # hook à l'inférence, α ∈ {-2, 0,5, 0,75, 1, 1,5, 2}
python run.py --set "pilotage.alphas=[-4, 4]"
```

## Ce que fait l'exemple

1. Extrait la direction « positif - négatif » sur 60 paires de phrases, à mi-profondeur.
2. Mesure le taux de textes jugés positifs sans intervention, sur des consignes neutres et sur des demandes de texte négatif.
3. **Mode vecteur** : la même mesure pour chaque α. Attendu : α > 0 augmente fortement le taux positif, même pour les demandes négatives ; α < 0 le fait chuter.
4. **Mode ablation** : attendu, les demandes de texte négatif produisent des textes nettement moins négatifs, avec des poids modifiés de façon permanente.

**Résultats** (Qwen2.5-1.5B-Instruct ; direction calculée sur 60 paires de phrases ; 100 demandes par type ; sujets de test jamais utilisés pour calculer la direction ; 3 graines ; écart avec la génération sans intervention, moyenne des 3 graines) :

| α | Demandes neutres : positivité | note du juge | perte par token | Demandes de texte négatif : positivité | note du juge | perte par token |
|---|---|---|---|---|---|---|
| sans intervention (valeur absolue) | 0,677 | −1,15 | 0,79 | 0,133 | −2,62 | 0,81 |
| 0,5 | +0,08 | −0,28 | +0,16 | +0,02 | −0,51 | +0,09 |
| 0,75 | +0,10 | −0,41 | +0,31 | +0,13 | −1,05 | +0,30 |
| 1,0 | +0,15 | −1,05 | +0,57 | +0,35 | −2,07 | +0,68 |
| 1,5 | +0,20 | −3,90 | +1,50 | +0,64 | −3,95 | +1,39 |
| 2,0 | +0,28 | −5,93 | +2,12 | +0,78 | −5,01 | +2,00 |
| −2,0 (concept opposé) | −0,57 | −3,10 | +1,68 | −0,04 | −1,81 | +1,09 |

La note du juge est celle d'un modèle de récompense externe (Skywork 0,6B) sur la réponse à la demande ; la perte par token est celle du texte généré sous le modèle non piloté (plus elle monte, moins le texte est cohérent). Avec 15 paires au lieu de 60, les écarts sont du même ordre (à α = 1 sur les demandes de texte négatif : positivité +0,40 contre +0,35, perte par token +0,68 dans les deux cas) : le nombre de paires ne change pas le compromis.

Le pilotage déplace bien le ton : la positivité monte avec α, et à α = 1 les demandes de texte négatif passent de 0,13 à 0,48. Mais la cohérence baisse en même temps. Le seuil de cohérence est fixé dans la configuration avant la mesure (`evaluation.seuil` : baisse de la note du juge ≤ 0,5 et hausse de la perte par token ≤ 0,2, sur les deux types de demande) et le rapport indique pour chaque α s'il le respecte (`verdict_seuil`). Aucun α ne satisfait à la fois ce seuil et une hausse nette de positivité sur les demandes de texte négatif, sur aucune des trois graines : à α = 0,5 la cohérence tient sur deux graines mais l'effet n'est pas net (+0,02) ; à α = 0,75 et au-delà la positivité monte nettement mais la cohérence ne tient plus. La variante reste « non prouvée » : c'est un réglage entre effet et cohérence, pas un gain gratuit.

**Profondeur d'injection** (`pilotage.couche_relative`, mêmes mesures sur les demandes de texte négatif, 3 graines ; écart avec la génération sans intervention ; dernière colonne : nombre de graines sur 3 où le seuil de cohérence est respecté avec une positivité nette) :

| Couche relative | α | Positivité | Note du juge | Perte par token | Seuil respecté |
|---|---|---|---|---|---|
| 0,25 | 0,5 | +0,07 | +0,05 | −0,04 | 0 / 3 (positivité non nette) |
| 0,25 | 1,0 | +0,05 | −0,08 | +0,07 | 0 / 3 |
| 0,4 | 0,5 | +0,09 | −0,21 | +0,06 | 1 / 3 |
| 0,4 | 1,0 | +0,47 | −1,70 | +0,55 | 0 / 3 |
| 0,5 (défaut) | 0,5 | +0,02 | −0,51 | +0,09 | 0 / 3 |
| 0,5 (défaut) | 1,0 | +0,35 | −2,07 | +0,68 | 0 / 3 |
| 0,6 | 0,5 | +0,16 | −0,65 | +0,17 | 1 / 3 |
| 0,6 | 1,0 | +0,46 | −2,14 | +0,65 | 0 / 3 |
| 0,75 | 0,5 | +0,05 | −0,24 | +0,08 | 1 / 3 |
| 0,75 | 1,0 | +0,12 | −0,64 | +0,26 | 0 / 3 |

À une couche très tôt (0,25), la direction n'a presque pas d'effet. Aux couches du milieu, l'effet sur la positivité apparaît avec le même coût en cohérence qu'à la couche par défaut. À aucune profondeur, aucun α (0,5, 0,75, 1,0) ne respecte le seuil de cohérence avec une positivité nette sur la majorité des graines.

**Mode ablation** (`pilotage.mode=ablation` : la direction est retirée des poids, sans hook à l'inférence ; 60 paires, 3 graines ; écart avec la génération sans intervention) :

| | Demandes de texte négatif | Demandes neutres |
|---|---|---|
| Positivité | **+0,32**, net sur chaque graine | +0,03, non significatif |
| Note du juge | −1,79 (p < 10⁻³) | −0,11, non significatif |
| Perte par token | −0,05, non significatif | +0,02, non significatif |

L'ablation déplace le ton des demandes de texte négatif sans dégrader la fluidité du texte (perte par token inchangée) et sans toucher aux demandes neutres. Le seuil de cohérence n'est pourtant pas respecté (`satisfait` = 0 sur les trois graines), à cause de la note du juge sur les demandes de texte négatif : le juge évalue la réponse à la demande, et un texte positif en réponse à « écris une plainte en colère » ne suit pas la consigne ; la baisse de la note mélange donc l'incohérence du texte et le non-respect de la consigne que le changement de ton produit par construction.

**Mesure de l'ablation sur trois graines neuves** (45, 46, 47 ; critère fixé avant la mesure : positivité d'au moins +0,2 avec un intervalle apparié qui exclut 0 sur les demandes de texte négatif, perte par token en hausse d'au plus 0,2 sur les deux types de demande, note du juge en baisse d'au plus 0,5 sur les demandes neutres seulement) :

| Graine | Positivité, texte négatif | Perte par token (négatif / neutre) | Note du juge, neutre |
|---|---|---|---|
| 45 | +0,29 [+0,17 ; +0,40] | −0,06 / +0,01 | −0,02 |
| 46 | +0,25 [+0,14 ; +0,36] | +0,02 / +0,03 | −0,04 |
| 47 | +0,35 [+0,25 ; +0,46] | −0,05 / +0,03 | +0,03 |

Le critère est satisfait sur chaque graine : l'ablation de la direction déplace le ton des demandes de texte négatif (+0,25 à +0,35 de positivité) sans dégrader la fluidité du texte ni les demandes neutres. Ce que cela démontre et ce que cela ne démontre pas : le critère diffère du seuil de cohérence par défaut (la note du juge n'est exigée que sur les demandes neutres), parce que ce seuil pénalise par construction un texte positif en réponse à « écris une plainte en colère » ; sur ces demandes, la note du juge baisse de 1,8 à 2,0, et un texte positif qui ne répond pas à la consigne n'est pas une amélioration pour qui l'avait demandée. Le pilotage par vecteur, lui, ne satisfait aucun des deux critères à aucun réglage essayé.

## Paramètres principaux (`config.yaml`)

| Clé | Rôle |
|---|---|
| `pilotage.mode` | `vecteur` ou `ablation` |
| `pilotage.couche` / `couche_relative` | Couche d'extraction et d'injection |
| `pilotage.alphas` | Intensités testées (multiples de l'écart moyen) |
| `pilotage.n_paires` | Nombre de paires contrastives |

## Grosse infra

- Inférence seulement : un 70B tient sur un nœud de 2 à 4 GPU 80 Go en `device_map=auto` ([`infra/config.yaml`](infra/config.yaml)). Pas d'entraînement distribué.
- En production ([`infra/lancer.sh`](infra/lancer.sh)) :
  - modèle ablaté servi comme n'importe quel checkpoint ;
  - vecteurs dynamiques via llama.cpp (`--control-vector`), repeng, EasySteer (vLLM) ;
  - pilotage par features de SAE (Gemma Scope, Llama Scope, SAELens).

## Pièges

- **Choix de la couche** : trop tôt, rien ne se passe ; trop tard, le texte se dégrade. Les couches centrales sont le bon point de départ.
- **Paires contrastives** : elles doivent ne différer *que* par le concept, sinon la direction capture autre chose (longueur, sujet).
- **α** : trouver le compromis effet / cohérence. Mesurer aussi une perplexité ou une tâche de contrôle.
- **Ablation** : très ciblée, mais irréversible sur le checkpoint. Garder l'original.
- **Usage** : la même technique sert à retirer des garde-fous de sécurité. Elle est présentée ici sur un concept neutre (le sentiment).

## Références

- Turner et al., *Activation Addition* (2023) · Rimsky et al., *Steering Llama 2 via Contrastive Activation Addition* (2024)
- Zou et al., *Representation Engineering* (2023) · Arditi et al., *Refusal in LLMs is Mediated by a Single Direction* (2024)
- Templeton et al., *Scaling Monosemanticity* (2024) · Gandikota et al., *Concept Sliders* (2023)
