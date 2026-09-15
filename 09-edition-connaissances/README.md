# 09 — Édition de connaissances (ROME, MEMIT, FT-L)

> Corriger ou modifier **un fait précis** stocké dans les poids (« la tour Eiffel est à Paris » → « … à Rome »), en touchant une seule matrice, sans réentraîner.

| Regroupe | Modifie | Local 12 Go | Test `--quick` |
|---|---|---|---|
| ROME, MEMIT (en infra), AlphaEdit, FT-L (fine-tuning contraint). Mentions : MEND, GRACE, WISE | Une matrice MLP (rang 1 par fait) | ✅ jusqu'à ~3B (ROME en fp32) | ✅ |

## Principe

Les travaux de *causal tracing* montrent que les faits « sujet → objet » sont en grande partie stockés dans les **MLP des premières couches intermédiaires**. La matrice de sortie d'un MLP (`down_proj`) agit comme une mémoire associative clé → valeur.

**ROME** ([`edition.py`](edition.py)) :

1. **Clé** `k*` : l'activation d'entrée de `down_proj` au dernier token du sujet, moyennée sur plusieurs contextes.
2. **Valeur** `v*` : on optimise un petit vecteur ajouté à la sortie du MLP, pour que le modèle prédise le nouvel objet. Une régularisation KL préserve le « sens » du sujet (« The Eiffel Tower is a… » ne doit pas changer).
3. **Mise à jour de rang 1** : `W' = W + (v* - W k*)(C⁻¹k*)ᵀ / ((C⁻¹k*)ᵀk*)`. La covariance `C` des clés sur du texte générique garantit que les *autres* clés sont le moins perturbées possible.

**MEMIT** généralise à des milliers de faits répartis sur plusieurs couches (infra). **AlphaEdit** projette la mise à jour dans l'espace qui ne touche pas les connaissances à préserver, ce qui améliore l'édition séquentielle. **FT-L** (`methode: ft_contraint`) fine-tune seulement cette matrice avec une contrainte de norme. C'est la référence simple, souvent moins spécifique.

Les métriques sont celles de CounterFact :

| Métrique | Question posée |
|---|---|
| **efficacité** | Le prompt édité donne-t-il le nouvel objet ? |
| **généralisation** | Les paraphrases aussi ? |
| **spécificité** | Les faits voisins (Louvre → Paris) sont-ils intacts ? |

## Modèles concernés

| Type de modèle | Applicable | Remarques |
|---|---|---|
| LLM décodeur (GPT-2, GPT-J, LLaMA, Qwen, Mistral...) | ✅ cas principal | Indiquer le module (`edition.module`) : `model.layers.{}.mlp.down_proj` pour LLaMA/Qwen/Mistral |
| GPT-2 (couches `Conv1D`) | ⚠️ | Poids transposés : adapter `rome()` (`W.T`) ou utiliser EasyEdit |
| MoE (Mixtral, Qwen-MoE) | ⚠️ | Le fait peut être réparti entre experts. Méthodes spécifiques ou mémoire externe (GRACE) |
| Encodeur (BERT), seq2seq (T5) | ⚠️ | MEND et KE ont été proposés pour ces architectures, moins courant |
| Multimodal (BLIP-2, LLaVA) | ⚠️ | EasyEdit propose l'édition multimodale (MMEdit) |
| Diffusion | ✅ | Même principe sur le cross-attention (TIME, UCE : « changer ce que signifie un mot ») |
| Modèle fermé via API | ❌ | Seul le RAG ([18](../18-rag)) permet de corriger un fait |

## Quand l'utiliser

- ✅ Quelques faits erronés ou obsolètes à corriger rapidement, sans pipeline d'entraînement.
- ✅ Recherche et interprétabilité : comprendre où et comment un fait est stocké.
- ⚠️ Des centaines d'éditions séquentielles dégradent le modèle : MEMIT/AlphaEdit en lot, ou mémoire externe.
- ❌ Les éditions **généralisent mal au raisonnement multi-sauts** (« la tour Eiffel est à Rome », donc « le pays de la tour Eiffel est » → souvent encore « France »). Pour des connaissances évolutives, préférer le RAG ([18](../18-rag)).

## Utilisation

```bash
python run.py --quick
python run.py                                         # ROME, 4 éditions successives sur Qwen2.5-1.5B
python run.py --set edition.methode=ft_contraint      # comparaison FT-L
python run.py --set edition.couche=12                 # éditer plus haut : spécificité et généralisation changent
python run.py --set modele.nom=meta-llama/Llama-3.2-1B
```

## Ce que fait l'exemple

1. Mesure efficacité, généralisation et spécificité sur 4 faits réels (Eiffel, Mona Lisa, Everest, Python), plus la complétion gloutonne du prompt.
2. Calcule `C` sur 200 paragraphes wikitext, puis applique les 4 éditions ROME l'une après l'autre.
3. Remesure. Attendu : efficacité et généralisation proches de 1, spécificité qui reste haute. Les complétions affichent les nouveaux objets.

## Paramètres principaux (`config.yaml`)

| Clé | Rôle |
|---|---|
| `edition.methode` | `rome` ou `ft_contraint` |
| `edition.module`, `couche` / `couche_relative` | Matrice éditée |
| `pas_valeur`, `lr_valeur`, `poids_kl`, `norme_max` | Optimisation de `v*` |
| `amortissement` | Stabilise l'inversion de `C` |

## Grosse infra

- ROME sur 7-8B : 1 GPU 40-80 Go ([`infra/config.yaml`](infra/config.yaml)).
- **MEMIT** sur 1 000 faits CounterFact avec [EasyEdit](https://github.com/zjunlp/EasyEdit) : [`infra/memit_easyedit.py`](infra/memit_easyedit.py) + [`infra/memit_llama3-8b.yaml`](infra/memit_llama3-8b.yaml). Le coût dominant est le calcul unique des statistiques de covariance (~100k échantillons Wikipédia).
- [`infra/lancer.sh`](infra/lancer.sh) : édition puis contrôle de non-régression (MMLU, HellaSwag).
- Pas besoin de multi-nœuds : l'édition reste une opération légère. Les modèles 70B demandent un nœud multi-GPU (`model_parallel: true`).

## Pièges

- **Mauvais module ou mauvaise couche** : l'édition « prend » sur le prompt exact mais ne généralise pas. Explorer 20 à 40 % de la profondeur.
- **Tokenisation de l'objet** : espace initial (`" Rome"`), objets de plusieurs tokens.
- **Éditions successives** : la norme cumulée des mises à jour explose au-delà de quelques dizaines. Surveiller la perplexité générale.
- **Évaluation trop indulgente** : toujours mesurer la spécificité et les paraphrases, pas seulement le prompt édité.

## Références

- Meng et al., *Locating and Editing Factual Associations in GPT (ROME)* (2022) · Meng et al., *MEMIT* (2023)
- Fang et al., *AlphaEdit* (2025) · Mitchell et al., *MEND* (2022) · Hartvigsen et al., *GRACE* (2023)
- Wang et al., *EasyEdit* (2024) · Cohen et al., *Evaluating the Ripple Effects of Knowledge Editing* (2024)
