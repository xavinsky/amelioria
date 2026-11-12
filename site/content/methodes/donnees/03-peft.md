---
title: PEFT (LoRA, QLoRA...)
id: peft
dossier: 03-peft
aka: Parameter-Efficient Fine-Tuning
schema:
  size: [760, 250]
  nodes:
    data: {label: Exemples annotés, sub: "question → réponse", external: true, x: 20, y: 24, w: 170}
    base: {label: Modèle de base, sub: "poids gelés", x: 20, y: 168, w: 170}
    train: {label: Entraînement, sub: "même boucle que le SFT", x: 290, y: 96, w: 190}
    adapt: {label: Adaptateur LoRA, sub: "matrices A et B, ~1 %", focus: true, x: 566, y: 24, w: 176}
    serve: {label: Service, sub: "une base, N adaptateurs", x: 566, y: 168, w: 176}
  flows:
    - {from: data, to: train, text: "Les exemples passent par la même boucle d'entraînement que le fine-tuning supervisé."}
    - {from: base, to: train, text: "Le modèle de base est chargé puis gelé : en bf16 pour LoRA, en 4 bits (NF4) pour QLoRA."}
    - {from: train, to: adapt, text: "Seules les petites matrices ajoutées à chaque couche linéaire reçoivent les gradients."}
    - {from: adapt, to: serve, text: "L'adaptateur, de quelques dizaines de Mo, est fusionné dans les poids ou chargé à chaud à côté d'autres adaptateurs."}
references:
  "Hu et al., LoRA: Low-Rank Adaptation of Large Language Models (2021)": https://arxiv.org/abs/2106.09685
  "Dettmers et al., QLoRA: Efficient Finetuning of Quantized LLMs (2023)": https://arxiv.org/abs/2305.14314
  "Liu et al., DoRA: Weight-Decomposed Low-Rank Adaptation (2024)": https://arxiv.org/abs/2402.09353
  "Kalajdzievski, A Rank Stabilization Scaling Factor for Fine-Tuning with LoRA (2023)": https://arxiv.org/abs/2312.03732
  "Meng et al., PiSSA (2024)": https://arxiv.org/abs/2404.02948
  "Liu et al., IA³ : Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper than In-Context Learning (2022)": https://arxiv.org/abs/2205.05638
  "Lester et al., The Power of Scale for Parameter-Efficient Prompt Tuning (2021)": https://arxiv.org/abs/2104.08691
  "Li & Liang, Prefix-Tuning (2021)": https://arxiv.org/abs/2101.00190
  "Biderman et al., LoRA Learns Less and Forgets Less (2024)": https://arxiv.org/abs/2405.09673
  "Ruiz et al., DreamBooth (2022)": https://arxiv.org/abs/2208.12242
  "Documentation Hugging Face PEFT": https://huggingface.co/docs/peft
---

On gèle le modèle et on n'entraîne qu'un petit module ajouté : 0,01 à 2 % des paramètres, pour une qualité proche du fine-tuning complet. L'adaptateur produit pèse quelques dizaines de Mo, se déploie à côté du modèle d'origine et se retire sans rien casser.

## Principe

Une couche linéaire calcule `y = W·x`. LoRA lui ajoute un chemin parallèle de rang faible : `y = W·x + (α/r)·B·A·x`, où `A` (r × d) et `B` (d × r) sont de petites matrices, `r` valant typiquement 8 à 64. `W` reste gelé ; `B` est initialisée à zéro, si bien que le modèle de départ est exactement le modèle d'origine. Seuls `A` et `B` apprennent, avec les mêmes données et la même perte que le [SFT](#lexterm-sft).

QLoRA applique la même idée à un modèle chargé en 4 bits (format NF4, voir [Quantification](#quantification)) : la mémoire est divisée par quatre environ, si bien qu'un modèle de 8 milliards de paramètres s'entraîne sur un GPU de 12 Go.

## Les variantes

Toutes suivent la même recette ; seul change le module ajouté. Dans le dépôt, on passe de l'une à l'autre par `SET="peft.methode=..."`.

| Méthode | Ce qui est entraîné | Particularité |
|---|---|---|
| **LoRA** | Deux matrices `B·A` sur chaque couche linéaire | Le standard. Adaptateur fusionnable dans les poids |
| **QLoRA** | LoRA sur un modèle chargé en 4 bits | Un 8B tient sur 12 Go, un 70B sur un GPU de 80 Go |
| **DoRA** | LoRA sur la direction des poids, plus une magnitude apprise | Souvent un peu meilleur que LoRA à rang égal, un peu plus lent |
| **rsLoRA** | LoRA avec une échelle `α/√r` | Stable quand on monte le rang (64 à 256) |
| **PiSSA** | LoRA initialisée sur les valeurs singulières principales de `W` | Converge plus vite |
| **LoHa, LoKr** | Produit de Hadamard ou de Kronecker au lieu de `B·A` | Courants pour les modèles de diffusion |
| **IA³** | Vecteurs qui multiplient les activations | Environ 0,01 % des paramètres |
| **Prompt tuning** | Embeddings de tokens virtuels placés avant le prompt | Le plus léger ; efficace surtout sur les gros modèles |
| **Prefix tuning** | Clés et valeurs virtuelles injectées dans chaque couche | Plus expressif que le prompt tuning |

## Quand l'utiliser

- Par défaut dès qu'on veut affiner un modèle : moins cher et plus rapide que le fine-tuning complet, et l'adaptateur se déploie facilement.
- QLoRA dès que le modèle en 16 bits ne tient pas en mémoire.
- Beaucoup de variantes d'un même modèle (une par client, par tâche) : un serveur comme vLLM charge plusieurs adaptateurs sur une seule base.
- Quand l'oubli compte : LoRA apprend moins que le fine-tuning complet, mais oublie aussi moins ce que le modèle savait faire.

**À éviter** pour faire entrer beaucoup de connaissances nouvelles ou changer fortement la distribution du modèle : le fine-tuning complet reste supérieur. Une LoRA de rang élevé (128 à 256) sur toutes les couches s'en rapproche. Pour des connaissances qui changent, le [RAG](#rag) est plus adapté.

## Pièges

- **Taux d'apprentissage** : celui d'un fine-tuning complet (1e-5) fait stagner une LoRA ; celui d'une LoRA (1e-4 à 2e-4) fait diverger un fine-tuning complet.
- **Couches ciblées** : une LoRA sur l'attention seule est nettement moins bonne que sur toutes les couches linéaires (`all-linear`).
- **Fusion d'une QLoRA** : ne pas fusionner dans le modèle 4 bits. Recharger la base en 16 bits, fusionner, puis requantifier si besoin.
- **Prompt et prefix tuning** sont faibles sur les modèles de moins d'un milliard de paramètres : c'est attendu.
