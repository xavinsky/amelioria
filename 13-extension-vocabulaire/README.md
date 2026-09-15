# 13 — Extension de vocabulaire

> Ajouter au tokenizer des tokens pour un domaine ou une langue mal couverts, agrandir les matrices d'embedding, initialiser les nouveaux vecteurs intelligemment, puis entraîner.

| Regroupe | Modifie | Local 12 Go | Test `--quick` |
|---|---|---|---|
| Ajout de tokens (added tokens), fusion de BPE, initialisation par moyenne des sous-tokens, entraînement en deux phases (embeddings, puis complet) | Tokenizer + taille des embeddings + poids | ✅ jusqu'à ~1B | ✅ |

## Principe

Un tokenizer entraîné sur du web anglais découpe mal un nom propre inventé, un terme de chimie ou le coréen : un mot devient 4 à 10 tokens. Les conséquences :

- séquences plus longues, donc **plus lentes et plus chères** ;
- contexte utile plus court ;
- apprentissage plus difficile.

Étapes de [`run.py`](run.py) :

1. **Choisir les tokens** : mots fréquents dans le corpus qui coûtent plusieurs tokens, classés par nombre de tokens économisés. Ou une liste explicite.
2. **Agrandir** : `resize_token_embeddings`. **Initialiser** chaque nouvel embedding à la *moyenne des embeddings de ses anciens sous-tokens*. Le modèle « comprend » ainsi déjà à peu près le nouveau token avant tout entraînement : la perte de départ est bien plus basse qu'avec une initialisation aléatoire (`vocabulaire.init=aleatoire` pour comparer).
3. **Phase 1** : n'entraîner que les embeddings d'entrée et de sortie, à lr élevé. Le reste du réseau, gelé, n'est pas perturbé par des vecteurs encore mal placés.
4. **Phase 2** : entraînement complet à lr faible. En pratique, c'est un pré-entraînement continu ([01](../01-pre-entrainement-continu)).

Mesures **comparables entre tokenizers** :

| Mesure | Ce qu'elle indique |
|---|---|
| `tokens_corpus_test` | Longueur du corpus de test en tokens : doit baisser |
| `bits_par_caractere` | Qualité de modélisation, indépendante du découpage. La perplexité par token ne se compare pas d'un tokenizer à l'autre |

Deux techniques d'ajout :

- **added tokens** (`run.py`) : simple, idéal pour quelques centaines de termes de domaine ou des tokens spéciaux (`<tool_call>`).
- **fusion de BPE** ([`infra/entrainer_tokenizer.py`](infra/entrainer_tokenizer.py)) : apprendre un BPE sur la langue cible et fusionner ses règles dans le tokenizer existant. C'est la bonne méthode pour une langue entière (milliers de tokens).

## Modèles concernés

| Type de modèle | Applicable | Remarques |
|---|---|---|
| LLM décodeur (toutes familles) | ✅ cas principal | Embeddings liés ou non liés gérés (`tie_word_embeddings`) |
| Encodeur (BERT, RoBERTa) / seq2seq (T5, mBART) | ✅ | Même procédure. Ajouter des termes médicaux à BERT est un classique |
| Modèle multimodal | ✅ | Ajout de tokens spéciaux (`<image>`, `<box>`) : c'est ainsi que les VLM sont construits |
| Speech (Whisper) | ⚠️ | Tokens de nouvelle langue possibles, peu courant |
| Tokenizer « byte-level » sans vocabulaire (ByT5, modèles à patchs) | ❌ | Rien à étendre |
| Modèle fermé via API | ❌ | |

## Quand l'utiliser

- ✅ Nouvelle langue ou écriture mal couverte (fertilité > 2 à 3 tokens par mot).
- ✅ Domaine très technique avec des termes récurrents : molécules, codes produits, identifiants.
- ✅ Nouveaux tokens de contrôle : outils, rôles, modalités.
- ⚠️ Toujours suivi d'un entraînement conséquent : sans lui, les nouveaux tokens sont moins bien compris que l'ancien découpage.
- ❌ Pour quelques mots rares : inutile, le modèle apprend très bien leur découpage.

## Utilisation

```bash
python run.py --quick
python run.py                                                  # Qwen2.5-0.5B + 200 noms fictifs du domaine
python run.py --set vocabulaire.init=aleatoire                 # comparer apres_init : bits par caractère bien plus hauts
python run.py --set "vocabulaire.liste=[Kavelmor, Zanlisia]"
python run.py --set donnees.source=hf --set donnees.nom=wikimedia/wikipedia --set donnees.config=20231101.br --set donnees.colonne=text --set donnees.n_textes=5000
```

## Ce que fait l'exemple

Sur une base de 80 personnages fictifs aux noms inventés, très mal découpés :

- `avant` : tokens et bits par caractère avec le tokenizer d'origine ;
- `apres_init` : ~200 tokens ajoutés, initialisés par moyenne. Le corpus est nettement plus court, avec des bits par caractère proches de `avant` ;
- `apres_phase1` puis `apres_phase2` : les bits par caractère descendent sous la valeur d'origine.

## Paramètres principaux (`config.yaml`)

| Clé | Rôle |
|---|---|
| `vocabulaire.n_nouveaux`, `min_freq`, `liste` | Choix des tokens |
| `vocabulaire.init` | `moyenne` ou `aleatoire` |
| `phase1.*` / `phase2.*` | Deux jeux d'hyperparamètres (lr élevé puis faible) |

## Grosse infra

- [`infra/entrainer_tokenizer.py`](infra/entrainer_tokenizer.py) : BPE de la langue cible fusionné dans le tokenizer d'origine (sur CPU, quelques minutes).
- [`infra/config.yaml`](infra/config.yaml) : Llama-3.1-8B + 10 000 tokens coréens, phase 1 (2 000 pas) et phase 2 sur 500k articles, FSDP sur 8×H100. Le vrai coût est la phase 2 : plusieurs milliards de tokens, voir [01 infra](../01-pre-entrainement-continu/infra).
- Méthodes avancées d'initialisation : FOCUS, WECHSEL (alignement d'embeddings multilingues), ZeTT (hypernetwork).

## Pièges

- **Taille multiple de 8 ou 64** : `resize_token_embeddings(pad_to_multiple_of=64)` accélère les GPU.
- **Embeddings liés** : sur Qwen2.5-0.5B, l'entrée et la sortie partagent la même matrice. Ne pas l'initialiser deux fois.
- **Espaces et added tokens** : un added token est extrait avant le BPE. Vérifier le découpage de « le mot » et de « lemot ».
- **Phase 1 sautée** : des gradients énormes sur des vecteurs mal placés déstabilisent tout le réseau.
- **Compatibilité** : un modèle au vocabulaire modifié n'est plus fusionnable avec ses « frères » ([08](../08-fusion-modeles)) ni utilisable en décodage spéculatif avec le draft d'origine.

## Références

- Cui et al., *Chinese-LLaMA* (2023) · Fujii et al., *Swallow* (2024) · Dobler & de Melo, *FOCUS* (2023) · Minixhofer et al., *WECHSEL* (2022)
- Hewitt, *Initializing New Word Embeddings for Pretrained Language Models* (2021)
