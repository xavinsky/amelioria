---
title: Quantification
id: quantification
dossier: 15-quantification
aka: Quantization (PTQ, QAT, GGUF)
schema:
  size: [760, 250]
  nodes:
    ref: {label: Modèle bf16, sub: "référence", x: 20, y: 24, w: 170}
    calib: {label: Textes de calibration, sub: "GPTQ, AWQ : 256 textes", external: true, x: 20, y: 168, w: 186}
    quant: {label: Quantification, sub: "NF4, int4, GPTQ, AWQ", focus: true, x: 290, y: 96, w: 180}
    mesure: {label: Mesure, sub: "mémoire, débit, qualité", x: 566, y: 24, w: 176}
    deploy: {label: Déploiement, sub: "vLLM, llama.cpp, Ollama", x: 566, y: 168, w: 176}
  flows:
    - {from: ref, to: quant, text: "Les poids des couches linéaires sont arrondis sur 8, 4 ou 3 bits, avec une échelle par groupe de 128 poids."}
    - {from: calib, to: quant, text: "GPTQ et AWQ mesurent les activations sur quelques centaines de textes représentatifs pour limiter l'erreur d'arrondi."}
    - {from: quant, to: mesure, from_offset: -12, text: "On compare au modèle bf16 dans les mêmes conditions : mémoire, tokens par seconde, perplexité, exactitude sur la tâche."}
    - {from: quant, to: deploy, from_offset: 12, text: "Le modèle quantifié est servi tel quel, ou exporté en GGUF pour llama.cpp et Ollama."}
references:
  "Dettmers et al., LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale (2022)": https://arxiv.org/abs/2208.07339
  "Dettmers et al., QLoRA, format NF4 (2023)": https://arxiv.org/abs/2305.14314
  "Frantar et al., GPTQ (2022)": https://arxiv.org/abs/2210.17323
  "Lin et al., AWQ: Activation-aware Weight Quantization (2023)": https://arxiv.org/abs/2306.00978
  "Xiao et al., SmoothQuant (2022)": https://arxiv.org/abs/2211.10438
  "Micikevicius et al., FP8 Formats for Deep Learning (2022)": https://arxiv.org/abs/2209.05433
  "Liu et al., LLM-QAT (2023)": https://arxiv.org/abs/2305.17888
  "PyTorch, Quantization-Aware Training for LLMs with torchao (2024)": https://pytorch.org/blog/quantization-aware-training/
  "llama.cpp : k-quants et matrice d'importance": https://github.com/ggml-org/llama.cpp
  "llm-compressor": https://github.com/vllm-project/llm-compressor
---

On stocke les poids sur 8, 4 ou 3 bits au lieu de 16. Le modèle devient 2 à 5 fois plus petit, souvent plus rapide, pour une perte de qualité qu'il faut mesurer. « Améliorer » veut dire ici rendre le modèle déployable : moins de mémoire, plus de débit, et parfois la possibilité de faire tourner un modèle plus grand sur le même matériel. Un 14B en 4 bits bat souvent un 7B en 16 bits.

## Principe

Les poids d'une couche sont découpés en groupes (souvent 128). Dans chaque groupe, on les remplace par des entiers sur quelques bits et une échelle commune. Les méthodes diffèrent par la façon de choisir ces entiers.

| Famille | Méthodes | Idée | Coût |
|---|---|---|---|
| **Sans calibration** | bitsandbytes NF4 et int8, torchao int8, int4, FP8 | Arrondi par groupe. NF4 place ses 16 niveaux selon une distribution normale, celle des poids | Quelques secondes, aucune donnée |
| **Avec calibration** | GPTQ, AWQ | Quelques centaines de textes servent à mesurer les activations. GPTQ corrige les erreurs d'arrondi colonne par colonne ; AWQ protège les canaux importants par une mise à l'échelle | Minutes à heures, 256 à 512 textes |
| **Pendant l'entraînement** | QAT | Pendant un court fine-tuning, les poids passent par une fausse quantification : le modèle apprend à tolérer l'arrondi | Un fine-tuning ; récupère une bonne part de la perte en 4 bits |
| **Format GGUF** | k-quants de llama.cpp | Blocs et super-blocs, matrice d'importance calculée sur un texte | Minutes, sur CPU |

**Poids seuls ou poids et activations.** W4A16 (poids 4 bits, calculs en 16 bits) réduit la mémoire et accélère la génération, limitée par la bande passante mémoire. W8A8 et FP8 accélèrent aussi les gros lots côté serveur, sur les GPU qui ont les noyaux adaptés.

## Quand l'utiliser

- Presque toujours au déploiement. En 8 bits la perte est quasi nulle ; en 4 bits elle est faible sur les modèles de 7 milliards de paramètres et plus.
- Pour faire tenir un modèle plus gros sur le matériel disponible.
- Pour entraîner au-dessus d'un modèle 4 bits : c'est [QLoRA](#peft).

**À éviter** sans mesure sur sa propre tâche en dessous de 4 bits, sur les petits modèles (moins de 3 milliards de paramètres) et pour les maths ou le code : les pertes y deviennent visibles. Pour réduire vraiment le nombre de paramètres, la distillation ou l'élagage vont plus loin.

## Pièges

- **Débit** : tous les formats ne sont pas plus rapides. bitsandbytes NF4 est souvent plus lent que bf16 en génération ; GPTQ, AWQ (noyaux Marlin de vLLM) et FP8 sont rapides.
- **Calibration non représentative** : calibrer un modèle de code sur des articles d'encyclopédie dégrade le résultat.
- **Perplexité trompeuse** : une petite hausse de perplexité peut cacher une chute nette sur les raisonnements longs. Toujours mesurer la tâche.
- **Format et moteur** doivent correspondre : compressed-tensors pour vLLM, GGUF pour llama.cpp et Ollama, bitsandbytes pour transformers.
- **GPTQ et AWQ dans transformers** : le modèle se charge compressé mais il est décompressé en 16 bits à la première génération. Le gain de mémoire et de débit se voit dans vLLM.
