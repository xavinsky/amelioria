# 15 — Quantification (bitsandbytes, torchao, GPTQ, AWQ, QAT, GGUF)

> Stocker les poids (et parfois les activations) sur 8, 4 ou 3 bits au lieu de 16 : un modèle 2 à 5 fois plus petit, souvent plus rapide, pour une perte de qualité mesurée.

| Regroupe | Modifie | Local 12 Go | Test `--quick` |
|---|---|---|---|
| PTQ sans calibration (bitsandbytes NF4/int8, torchao int8/int4/fp8), PTQ avec calibration (GPTQ, AWQ), QAT (quantization-aware training), GGUF k-quants + imatrix (llama.cpp) | Représentation des poids | ✅ jusqu'à ~8B à quantifier, 70B en GGUF/GPTQ couche par couche | ✅ (`torchao_int8wo`, CPU) |

## Principe

Schéma : [docs/schemas-methodes.md](../docs/schemas-methodes.md#m15) (ce que la méthode ajoute ou modifie dans le transformer).

« Améliorer » veut dire ici rendre le modèle **déployable** : moins de mémoire, plus de débit, et parfois la possibilité de faire tourner un modèle *plus grand* sur le même matériel. Un 14B en 4 bits bat souvent un 7B en 16 bits.

| Famille | Méthodes | Idée | Coût |
|---|---|---|---|
| **PTQ à la volée** | `bnb_nf4`, `bnb_int8`, `torchao_int8wo`, `torchao_int4wo`, `torchao_fp8` | Arrondi par groupes de poids, avec une échelle par groupe. NF4 utilise des niveaux adaptés à une distribution normale | Secondes, pas de données |
| **PTQ calibrée** | `gptq`, `awq` | Quelques centaines d'exemples servent à mesurer les activations. GPTQ corrige les erreurs d'arrondi colonne par colonne (inverse de la hessienne) ; AWQ protège les canaux importants par une mise à l'échelle | Minutes à heures, 256 à 512 exemples |
| **QAT** | `qat` | Pendant un court fine-tuning, les poids passent par une *fausse* quantification : le modèle apprend à tolérer l'arrondi, puis conversion réelle | Un fine-tuning. Récupère une bonne partie de la perte de la PTQ en 4 bits et moins |
| **GGUF** | [`variantes/gguf-llamacpp`](variantes/gguf-llamacpp) | k-quants à blocs et super-blocs, matrice d'importance (imatrix) calculée sur un texte | Minutes sur CPU |

Mesures (référence bf16 puis modèle quantifié) : mémoire du modèle, tokens/s en génération, perplexité wikitext, exactitude sur la tâche.

**Poids seuls ou poids + activations** : W4A16 réduit la mémoire et accélère le décodage, limité par la bande passante mémoire. W8A8 et FP8 accélèrent les gros lots (prefill, serveur) sur les GPU qui ont les noyaux adaptés.

## Modèles concernés

| Type de modèle | Applicable | Remarques |
|---|---|---|
| LLM décodeur (toutes familles) | ✅ cas principal | [`run.py`](run.py) |
| MoE (Mixtral, Qwen-MoE, DeepSeek) | ✅ | Experts quantifiés, routeur souvent laissé en haute précision |
| Encodeurs (BERT) | ✅ | int8 dynamique (`torch.ao`, ONNX Runtime, Optimum) très courant en CPU |
| Vision / VLM | ✅ | Tour vision souvent gardée en 16 bits, LLM quantifié |
| Diffusion (SDXL, Flux) | ✅ | NF4/FP8 du transformer de diffusion (diffusers + bitsandbytes/torchao), GGUF pour Flux |
| Whisper | ✅ | int8 (CTranslate2 / faster-whisper), GGUF (whisper.cpp) |
| Modèle fermé via API | ❌ | Le fournisseur le fait, ou pas |

## Quand l'utiliser

- ✅ Toujours envisager en déploiement. 8 bits : perte quasi nulle ; 4 bits : perte faible sur les modèles ≥ 7B.
- ✅ Faire tenir un modèle plus gros sur le matériel disponible.
- ✅ QLoRA ([03](../03-peft)) : entraîner au-dessus d'un modèle NF4.
- ⚠️ Moins de 4 bits, petits modèles (< 3B), maths et code : les pertes deviennent visibles. Toujours mesurer sur *sa* tâche, pas seulement la perplexité.

## Utilisation

```bash
python run.py --quick                                      # torchao int8 poids seuls (fonctionne sur CPU)
python run.py                                              # NF4 bitsandbytes, Qwen2.5-1.5B-Instruct
../.venv-llmcompressor/bin/python run.py --set methode=gptq   # llmcompressor, dans son propre venv
../.venv-llmcompressor/bin/python run.py --set methode=awq
python run.py --set methode=torchao_int4wo
python run.py --set methode=qat --set modele.nom=Qwen/Qwen2.5-0.5B-Instruct   # fine-tuning complet fp32 : 0,5B sur 12 Go
python variantes/gguf-llamacpp/run.py --quick              # nécessite llama.cpp compilé (voir config)
```

Dépendances selon la méthode : `bitsandbytes` (bnb), `torchao` (torchao, qat), `llmcompressor` (gptq, awq). llm-compressor impose sa propre version de torch et de transformers : il s'installe dans un venv séparé (`python3 -m venv .venv-llmcompressor && .venv-llmcompressor/bin/pip install llmcompressor datasets pyyaml`). Voir [`requirements-extras.txt`](../requirements-extras.txt).

## Paramètres principaux (`config.yaml`)

| Clé | Rôle |
|---|---|
| `methode` | Voir le tableau |
| `gptq.schema` | `W4A16`, `W8A16` ou `W8A8` |
| `calibration.n`, `longueur` | Données de calibration (GPTQ/AWQ) : représentatives de l'usage réel |
| `qat.*` | Groupe de quantification et fine-tuning QAT |

## Grosse infra

- **Quantifier ne demande pas de cluster.** llm-compressor calibre un 70B couche par couche sur 1 à 2 GPU ([`infra/gptq_70b_llmcompressor.py`](infra/gptq_70b_llmcompressor.py)). GGUF tourne sur CPU.
- Le multi-GPU sert à **servir** (vLLM `--tensor-parallel-size`) et à **évaluer la dégradation** sur des benchmarks ([`infra/lancer.sh`](infra/lancer.sh) : lm-eval du modèle bf16 et du modèle quantifié).
- [`infra/config.yaml`](infra/config.yaml) : `run.py` complet sur un 8B (GPTQ W4A16) sur 1 GPU 40 à 80 Go.
- QAT sur gros modèles : torchtune / Axolotl (recettes QAT torchao), avec la même infra que le SFT ([02](../02-fine-tuning-supervise)).
- Cache KV quantifié (FP8) pour les longs contextes : `vllm serve ... --kv-cache-dtype fp8`.

## Pièges

- **Débit** : tous les formats ne sont pas plus rapides. bitsandbytes NF4 est souvent *plus lent* que bf16 en génération. GPTQ/AWQ (noyaux Marlin dans vLLM) et FP8 sont rapides.
- **Calibration non représentative** : du texte wikitext pour un modèle de code donne un résultat dégradé.
- **Perplexité trompeuse** : une petite hausse de perplexité peut masquer une chute nette sur des raisonnements longs.
- **Couches sensibles** : garder `lm_head` (et parfois les embeddings, le routeur MoE) en haute précision.
- **Format** et moteur d'inférence doivent correspondre : compressed-tensors pour vLLM, GGUF pour llama.cpp / Ollama, bnb pour transformers.
- **GPTQ/AWQ dans transformers** : le modèle se charge compressé, mais transformers le décompresse en bf16 dès la première génération (pas de noyaux 4 bits compressed-tensors). Le gain de mémoire et de débit se voit dans vLLM. `run.py` mesure donc la mémoire avant de générer, et ajoute la taille sur disque.
- **Format int4 et noyaux** : un format quantifié n'est utile que si des noyaux l'exécutent sur le matériel visé. `run.py` utilise le format torchao `tile_packed_to_4d`, dont les noyaux int4 (tinygemm) sont inclus dans PyTorch CUDA.

## Références

- Dettmers et al., *LLM.int8()* (2022) et *QLoRA / NF4* (2023) · Frantar et al., *GPTQ* (2022) · Lin et al., *AWQ* (2023)
- Xiao et al., *SmoothQuant* (2022) · Liu et al., *LLM-QAT* (2023) · PyTorch, *torchao QAT* (2024)
- llama.cpp k-quants et imatrix : https://github.com/ggml-org/llama.cpp · llm-compressor : https://github.com/vllm-project/llm-compressor
