# 08 — Fusion de modèles (model merging)

> Additionner intelligemment les poids de plusieurs fine-tunes d'un même modèle pour obtenir un modèle qui cumule leurs compétences, **sans entraînement ni GPU**.

| Regroupe | Modifie | Local 12 Go | Test `--quick` |
|---|---|---|---|
| Model soups (moyenne), SLERP, task arithmetic, TIES, DARE (+TIES), fusion évolutionnaire | Tous les poids (combinaison) | ✅ fusion sur CPU/RAM, jusqu'à ~7B avec 64 Go de RAM, 70B+ avec mergekit en mode lazy | ✅ |

## Principe

Schéma : [docs/schemas-methodes.md](../docs/schemas-methodes.md#m08) (ce que la méthode ajoute ou modifie dans le transformer).

Deux fine-tunes d'un même modèle de base restent dans le même « bassin » de l'espace des poids. Leur combinaison donne souvent un modèle qui fonctionne, et qui hérite des deux.

On appelle **vecteur de tâche** la différence `τ_i = θ_i - θ_base` : ce que le fine-tuning i a ajouté.

| `methode` | Formule / idée | Base requise |
|---|---|---|
| `lineaire` | Moyenne pondérée des poids (model soup). Idéal pour des variantes d'un même entraînement | non |
| `slerp` | Interpolation sphérique entre **deux** modèles. Garde mieux la norme des poids | non |
| `arithmetique_taches` | `θ = base + Σ λ_i · τ_i`. On peut aussi *soustraire* une tâche (λ < 0) pour retirer un comportement | oui |
| `ties` | 1) garder les coordonnées de plus forte magnitude de chaque `τ_i` ; 2) élire un signe par coordonnée ; 3) moyenner seulement les valeurs d'accord. Réduit les interférences | oui |
| `dare_ties` | Comme TIES, mais on abandonne les coordonnées **au hasard** et on rééchelonne (`1/densité`) | oui |
| `dare_lineaire` | DARE puis arithmétique des tâches | oui |
| évolutionnaire | Un algorithme évolutionnaire cherche les poids et densités par couche qui maximisent un score (voir l'infra) | oui |

Les implémentations sont dans [`fusion.py`](fusion.py) : environ 100 lignes lisibles, en PyTorch pur, applicables tenseur par tenseur à n'importe quelle architecture.

## Modèles concernés

| Type de modèle | Applicable | Remarques |
|---|---|---|
| LLM décodeur : fine-tunes d'un **même** modèle de base | ✅ cas principal | Mêmes formes de tenseurs. Les fusions entre familles différentes (Llama + Qwen) ne marchent pas |
| Adaptateurs LoRA d'une même base | ✅ | `peft` : `model.add_weighted_adapter([...], weights, combination_type="ties")` |
| Encodeurs, ViT, CLIP, Whisper | ✅ | Model soups sur CLIP/ViT (papier d'origine), WiSE-FT (fusion modèle fine-tuné / zéro-shot) |
| Diffusion (SD, SDXL) | ✅ très courant | Fusion de checkpoints et de LoRA de style (outils : Automatic1111, ComfyUI, `sd-mecha`) |
| Modèles de tailles ou d'architectures différentes | ❌ | Il faut passer par la distillation ([06](../06-distillation)) |
| Modèle fermé via API | ❌ | Pas d'accès aux poids |

## Quand l'utiliser

- ✅ Plusieurs fine-tunes spécialisés et un seul modèle à déployer : combiner code, maths et instruction.
- ✅ Récupérer des capacités générales perdues pendant un fine-tuning : fusionner le modèle spécialisé avec l'instruct d'origine (WiSE-FT, contre l'oubli).
- ✅ Retirer un comportement : soustraire un vecteur de tâche.
- ✅ Budget GPU nul : quelques minutes de CPU.
- ⚠️ Toujours évaluer : une fusion peut dégrader silencieusement une compétence.

## Utilisation

```bash
python run.py --quick                                                  # moyenne SmolLM2 base + instruct
python run.py                                                          # TIES : Qwen2.5-1.5B-Instruct + Qwen2.5-Math-1.5B-Instruct
python run.py --set methode=dare_ties --set densite=0.3
python run.py --set methode=slerp --set base=null --set t_slerp=0.3
python run.py --set methode=arithmetique_taches
```

## Ce que fait l'exemple

1. Évalue chaque modèle source sur GSM8K (100 questions) et sur la perplexité générale.
2. Fusionne avec la méthode choisie, par rapport à la base commune Qwen2.5-1.5B.
3. Évalue le modèle fusionné. Attendu avec TIES : un score GSM8K proche du modèle Math, tout en gardant le format et la perplexité générale de l'instruct.

## Paramètres principaux (`config.yaml`)

| Clé | Rôle |
|---|---|
| `methode` | Voir le tableau |
| `base` | Modèle d'origine commun (`null` pour `lineaire` et `slerp`) |
| `modeles[].poids` | λ de chaque source. Une somme > 1 est courante avec TIES et DARE |
| `densite` | TIES/DARE : part conservée de chaque vecteur de tâche (0,2 à 0,7) |
| `t_slerp` | Position entre les deux modèles |

## Grosse infra

- Pas besoin de cluster pour **fusionner** : [mergekit](https://github.com/arcee-ai/mergekit) fusionne un 70B avec ~32 Go de RAM en chargeant les tenseurs un par un (`--lazy-unpickle`).
- [`infra/ties_70b.yml`](infra/ties_70b.yml) : TIES entre trois fine-tunes de Llama-3.1-70B.
- [`infra/slerp_par_couche.yml`](infra/slerp_par_couche.yml) : SLERP dont le coefficient varie par couche et par type (attention / MLP).
- [`infra/evolve.yml`](infra/evolve.yml) + `mergekit-evolve` : **fusion évolutionnaire**. Des centaines de candidats, chacun évalué : c'est là que les GPU sont nécessaires (vLLM, plusieurs nœuds via Ray).
- [`infra/config.yaml`](infra/config.yaml) : `run.py` sur trois modèles 7B (DARE-TIES).

## Pièges

- **Base commune obligatoire** pour les méthodes à vecteurs de tâche : vérifier l'ascendance des modèles. Le nom ne suffit pas : Qwen2.5-Math-1.5B-Instruct et Qwen2.5-Coder-1.5B-Instruct viennent de Qwen2.5-1.5B, mais après des centaines de milliards de tokens de pré-entraînement continu (et un `rope_theta` différent pour Math). Leurs poids sont à 100 % et 61 % de la base, contre 0,8 % pour Qwen2.5-1.5B-Instruct : leur fusion avec l'Instruct détruit le modèle. `run.py` affiche cet écart relatif pour chaque source (`ecart_relatif_a_la_base`).
- **Vocabulaire étendu** : les embeddings de tailles différentes ne se fusionnent pas. Ici, ceux du premier modèle sont conservés.
- **Chat templates différents** : choisir explicitement le tokenizer (`tokenizer`).
- **Contamination** des classements : beaucoup de fusions « record » surapprennent les benchmarks. Évaluer sur ses propres tâches.

## Références

- Wortsman et al., *Model Soups* (2022) · Ilharco et al., *Editing Models with Task Arithmetic* (2023)
- Yadav et al., *TIES-Merging* (2023) · Yu et al., *DARE: Language Models are Super Mario* (2024)
- Akiba et al., *Evolutionary Optimization of Model Merging Recipes* (Sakana AI, 2024) · Goddard et al., *mergekit* (2024)
