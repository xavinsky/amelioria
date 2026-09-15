# 02 — Fine-tuning supervisé (SFT)

> Entraîner le modèle sur des exemples « entrée -> sortie idéale » pour lui apprendre une tâche, un format ou un comportement.

| Regroupe | Modifie | Local 12 Go | Test `--quick` |
|---|---|---|---|
| SFT, instruction tuning, fine-tuning complet, fine-tuning de classifieur, de modèle vision / seq2seq / speech | Tous les poids | ✅ jusqu'à ~1B (LLM), tous les encodeurs / ViT / Whisper-small | ✅ |

## Principe

On minimise la perte d'entropie croisée entre la sortie du modèle et la sortie attendue. Pour un LLM, la perte n'est calculée que sur les tokens de la réponse (`completion_only_loss`) : le modèle apprend à répondre, pas à recopier la question.

Le fine-tuning *complet* met à jour tous les poids. C'est la version la plus expressive, mais aussi la plus coûteuse en mémoire. Avec AdamW en précision mixte, il faut compter ~16 octets par paramètre : poids fp32, gradients, deux moments, plus les activations. Pour faire la même chose avec 10 fois moins de mémoire, voir LoRA et QLoRA ([03](../03-peft)). La recette de données est identique.

Ce que le SFT apprend bien : le format, le style, la langue de réponse, une tâche précise, l'utilisation d'outils, le raisonnement pas à pas démontré dans les exemples. Ce qu'il apprend mal : des connaissances nouvelles fiables (risque d'hallucination accru), et un comportement meilleur que celui des exemples eux-mêmes.

## Modèles concernés

| Type de modèle | Applicable | Exemple dans ce dossier |
|---|---|---|
| LLM décodeur (LLaMA, Qwen, Mistral, Gemma, Phi, SmolLM, GPT-2...) | ✅ | [`run.py`](run.py) : SFT sur problèmes à réponse vérifiable |
| Encodeur (BERT, RoBERTa, ModernBERT, DeBERTa, CamemBERT) | ✅ | [`variantes/classification-encodeur`](variantes/classification-encodeur) : sentiment SST-2 |
| Encodeur-décodeur (T5, FLAN-T5, BART, mT5, NLLB) | ✅ | [`variantes/seq2seq`](variantes/seq2seq) |
| Vision (ViT, DeiT, DINOv2, ConvNeXt, Swin) | ✅ | [`variantes/vision`](variantes/vision) : CIFAR-10 |
| Audio / speech (Whisper, wav2vec2) | ✅ | [`variantes/audio-whisper`](variantes/audio-whisper) : français, mesure du WER |
| Multimodal (Qwen2.5-VL, LLaVA, Idefics) | ✅ | Même `SFTTrainer`, avec des messages contenant des images (voir la doc TRL « VLM SFT ») |
| Diffusion (Stable Diffusion, Flux) | ✅ | Fine-tuning complet ou DreamBooth : voir [03 variante diffusion](../03-peft/variantes/diffusion-lora), en passant de LoRA à `train_text_to_image.py` |
| Modèle fermé via API | ⚠️ | Fine-tuning « as a service » (OpenAI, Vertex AI Gemini, Bedrock) : on envoie un JSONL au même format prompt/réponse, sans accès aux poids |

## Quand l'utiliser

- ✅ Un format ou un comportement précis, reproductible, avec quelques centaines à quelques dizaines de milliers d'exemples.
- ✅ Transformer un modèle de base en assistant (instruction tuning).
- ✅ Distiller les réponses d'un gros modèle vers un petit ([06](../06-distillation), [07](../07-donnees-synthetiques)).
- ❌ Si le prompting suffit ([17](../17-optimisation-prompt)) : toujours essayer avant.
- ❌ Pour des connaissances évolutives : RAG ([18](../18-rag)).
- ⚠️ Si la mémoire manque : LoRA ou QLoRA ([03](../03-peft)), en gardant exactement les mêmes données.

## Utilisation

```bash
python run.py --quick                                        # SmolLM2-135M-Instruct, 64 exemples
python run.py                                                # Qwen2.5-0.5B-Instruct, 3000 exemples, ~10 min
python run.py --set donnees.source=gsm8k --set donnees.n_train=7473
python run.py --set modele.nom=meta-llama/Llama-3.2-1B-Instruct --set entrainement.gradient_checkpointing=true
python variantes/classification-encodeur/run.py --quick
python variantes/vision/run.py --quick
python variantes/seq2seq/run.py --quick
python variantes/audio-whisper/run.py --quick
```

## Ce que fait l'exemple

1. Génère des problèmes `(a op b) op c` avec leur raisonnement pas à pas et leur réponse finale (`Answer: N`).
2. Mesure l'exactitude du modèle (réponse finale extraite) sur 200 problèmes jamais vus.
3. SFT complet sur 3000 problèmes, perte uniquement sur la réponse.
4. Remesure. Attendu avec Qwen2.5-0.5B-Instruct : un gain net d'exactitude et un format de réponse devenu constant.

## Paramètres principaux (`config.yaml`)

| Clé | Rôle |
|---|---|
| `modele.nom`, `modele.dtype` | Modèle, et `float32` pour des poids maîtres en fp32 (calcul bf16) |
| `donnees.source` | `arith`, `gsm8k` ou `hf` (+ `nom`, `champs`) |
| `entrainement.lr` | 5e-6 à 2e-5 en SFT complet (10 fois plus en LoRA) |
| `entrainement.max_length` | Longueur max prompt + réponse |
| `evaluation.mode` | `nombre` (réponse numérique) ou `contient` |

**Conversations multi-tours** (ex. `HuggingFaceTB/smoltalk`) : `SFTTrainer` accepte directement une colonne `messages` (liste `{role, content}`). Il suffit de remplacer `entrainement.exemples_sft(...)` par `[{"messages": m} for m in ds["messages"]]` et de passer `assistant_only_loss=True`.

## Grosse infra

- Local jusqu'à ~1B de paramètres en SFT complet (Qwen2.5-0.5B : ~9 Go ; 1,5B demande gradient checkpointing et un optimiseur 8 bits).
- [`infra/config.yaml`](infra/config.yaml) : Llama-3.1-8B-Instruct sur GSM8K, FSDP2 sur 8×H100. Pour 70B : 2 à 4 nœuds (même config, `sbatch commun/infra/slurm_multinoeud.sbatch ...`).
- [`infra/lancer.sh`](infra/lancer.sh) : entraînement puis évaluation lm-eval + vLLM.
- Frameworks qui encapsulent la même recette : `trl sft` (CLI TRL), [Axolotl](https://github.com/axolotl-ai-cloud/axolotl), [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory), [torchtune](https://github.com/pytorch/torchtune), NeMo.

## Pièges

- **Chat template** : entraîner et servir avec le même template (le `SFTTrainer` utilise celui du tokenizer).
- **Surapprentissage** : sur de petits jeux, 1 à 3 epochs suffisent. Surveiller une métrique de généralisation.
- **Oubli** : le SFT étroit dégrade les capacités générales. Mélanger des données générales ou passer par LoRA ([docs/oubli-catastrophique.md](../docs/oubli-catastrophique.md)).
- **Qualité plutôt que quantité** : 1 000 exemples excellents valent mieux que 100 000 médiocres (LIMA).
- **T5 en fp16/bf16** : instable, rester en fp32 (voir la config seq2seq).

## Références

- Ouyang et al., *InstructGPT* (2022) : étape SFT.
- Zhou et al., *LIMA: Less Is More for Alignment* (2023).
- Documentation TRL `SFTTrainer` : https://huggingface.co/docs/trl/sft_trainer
