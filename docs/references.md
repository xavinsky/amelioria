# Références

Papiers de recherche des techniques du dépôt, rangés par dossier. Les identifiants arXiv renvoient à la page du papier (année de première soumission). Les schémas de chaque technique sont dans [schemas-methodes.md](schemas-methodes.md).

## Fondations du transformer

| Technique | Papier | Lien | Ce qu'il apporte |
|---|---|---|---|
| **Transformer** | Vaswani et al., *Attention Is All You Need* (2017) | [1706.03762](https://arxiv.org/abs/1706.03762) | Architecture à base d'attention, sans récurrence |
| **BPE** | Sennrich et al., *Neural Machine Translation of Rare Words with Subword Units* (2015) | [1508.07909](https://arxiv.org/abs/1508.07909) | Découpage en sous-mots des tokenizers modernes |
| **RoPE** | Su et al., *RoFormer: Enhanced Transformer with Rotary Position Embedding* (2021) | [2104.09864](https://arxiv.org/abs/2104.09864) | Encodage de la position par rotation de q et k |
| **RMSNorm** | Zhang & Sennrich, *Root Mean Square Layer Normalization* (2019) | [1910.07467](https://arxiv.org/abs/1910.07467) | Normalisation utilisée par LLaMA, Qwen, Mistral |
| **SwiGLU** | Shazeer, *GLU Variants Improve Transformer* (2020) | [2002.05202](https://arxiv.org/abs/2002.05202) | MLP à porte des LLM actuels (gate, up, down) |
| **GQA** | Ainslie et al., *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints* (2023) | [2305.13245](https://arxiv.org/abs/2305.13245) | Têtes de clés/valeurs partagées : cache KV réduit |
| **FlashAttention** | Dao et al., *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness* (2022) | [2205.14135](https://arxiv.org/abs/2205.14135) | Attention exacte, rapide et économe en mémoire |
| **Lois d'échelle** | Kaplan et al., *Scaling Laws for Neural Language Models* (2020) | [2001.08361](https://arxiv.org/abs/2001.08361) | Relation entre taille, données, calcul et perte |
| **LLaMA** | Touvron et al., *LLaMA: Open and Efficient Foundation Language Models* (2023) | [2302.13971](https://arxiv.org/abs/2302.13971) | Recette du LLM décodeur ouvert de référence |
| **BERT** | Devlin et al., *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding* (2018) | [1810.04805](https://arxiv.org/abs/1810.04805) | Encodeur bidirectionnel pré-entraîné par MLM |
| **RoBERTa** | Liu et al., *RoBERTa: A Robustly Optimized BERT Pretraining Approach* (2019) | [1907.11692](https://arxiv.org/abs/1907.11692) | BERT mieux entraîné (encodeur des exemples) |
| **T5** | Raffel et al., *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer* (2019) | [1910.10683](https://arxiv.org/abs/1910.10683) | Encodeur-décodeur texte vers texte |
| **ViT** | Dosovitskiy et al., *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale* (2020) | [2010.11929](https://arxiv.org/abs/2010.11929) | Transformer sur des patchs d'image |
| **Whisper** | Radford et al., *Robust Speech Recognition via Large-Scale Weak Supervision* (2022) | [2212.04356](https://arxiv.org/abs/2212.04356) | Reconnaissance vocale encodeur-décodeur |
| **DDPM** | Ho et al., *Denoising Diffusion Probabilistic Models* (2020) | [2006.11239](https://arxiv.org/abs/2006.11239) | Modèles de diffusion par débruitage |
| **Latent Diffusion** | Rombach et al., *High-Resolution Image Synthesis with Latent Diffusion Models* (2021) | [2112.10752](https://arxiv.org/abs/2112.10752) | Diffusion dans l'espace latent d'un VAE (Stable Diffusion) |
| **Switch Transformer** | Fedus et al., *Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity* (2021) | [2101.03961](https://arxiv.org/abs/2101.03961) | Mixture d'experts creuse à grande échelle |

## Entraîner avec peu de mémoire

| Technique | Papier | Lien | Ce qu'il apporte |
|---|---|---|---|
| **Adam** | Kingma & Ba, *Adam: A Method for Stochastic Optimization* (2014) | [1412.6980](https://arxiv.org/abs/1412.6980) | Optimiseur à moments adaptatifs |
| **AdamW** | Loshchilov & Hutter, *Decoupled Weight Decay Regularization* (2017) | [1711.05101](https://arxiv.org/abs/1711.05101) | Weight decay découplé, optimiseur par défaut |
| **Précision mixte** | Micikevicius et al., *Mixed Precision Training* (2017) | [1710.03740](https://arxiv.org/abs/1710.03740) | Poids maîtres fp32, calculs en demi-précision |
| **Gradient checkpointing** | Chen et al., *Training Deep Nets with Sublinear Memory Cost* (2016) | [1604.06174](https://arxiv.org/abs/1604.06174) | Recalculer les activations au lieu de les stocker |
| **Optimiseurs 8 bits** | Dettmers et al., *8-bit Optimizers via Block-wise Quantization* (2021) | [2110.02861](https://arxiv.org/abs/2110.02861) | États d'Adam quantifiés par blocs (adamw_bnb_8bit) |
| **ZeRO** | Rajbhandari et al., *ZeRO: Memory Optimizations Toward Training Trillion Parameter Models* (2019) | [1910.02054](https://arxiv.org/abs/1910.02054) | Répartir poids, gradients et états d'optimiseur sur plusieurs GPU |

## 01 — Pré-entraînement continu

| Technique | Papier | Lien | Ce qu'il apporte |
|---|---|---|---|
| **DAPT / TAPT** | Gururangan et al., *Don't Stop Pretraining: Adapt Language Models to Domains and Tasks* (2020) | [2004.10964](https://arxiv.org/abs/2004.10964) | Poursuivre le pré-entraînement sur le domaine ou la tâche |
| **Replay et re-warming** | Ibrahim et al., *Simple and Scalable Strategies to Continually Pre-train Large Language Models* (2024) | [2403.08763](https://arxiv.org/abs/2403.08763) | Lr, warmup et replay pour pré-entraîner en continu sans oubli |
| **Paraphrases** | Allen-Zhu & Li, *Physics of Language Models: Part 3.1, Knowledge Storage and Extraction* (2023) | [2309.14316](https://arxiv.org/abs/2309.14316) | Plusieurs formulations d'un fait pour qu'il soit réutilisable en question-réponse |

## 02 — Fine-tuning supervisé

| Technique | Papier | Lien | Ce qu'il apporte |
|---|---|---|---|
| **Instruction tuning (FLAN)** | Wei et al., *Finetuned Language Models Are Zero-Shot Learners* (2021) | [2109.01652](https://arxiv.org/abs/2109.01652) | Fine-tuning multi-tâches formulé en instructions |
| **InstructGPT** | Ouyang et al., *Training language models to follow instructions with human feedback* (2022) | [2203.02155](https://arxiv.org/abs/2203.02155) | SFT puis RLHF : la recette des assistants |
| **LIMA** | Zhou et al., *LIMA: Less Is More for Alignment* (2023) | [2305.11206](https://arxiv.org/abs/2305.11206) | Peu d'exemples de grande qualité suffisent à l'alignement du format |
| **banking77** | Casanueva et al., *Efficient Intent Detection with Dual Sentence Encoders* (2020) | [2003.04807](https://arxiv.org/abs/2003.04807) | Jeu de 77 intentions de clients bancaires, référence de la détection d'intentions |

## 03 — PEFT

| Technique | Papier | Lien | Ce qu'il apporte |
|---|---|---|---|
| **LoRA** | Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models* (2021) | [2106.09685](https://arxiv.org/abs/2106.09685) | Mise à jour de rang faible B·A ajoutée aux couches linéaires |
| **QLoRA** | Dettmers et al., *QLoRA: Efficient Finetuning of Quantized LLMs* (2023) | [2305.14314](https://arxiv.org/abs/2305.14314) | LoRA sur un modèle NF4 4 bits |
| **DoRA** | Liu et al., *DoRA: Weight-Decomposed Low-Rank Adaptation* (2024) | [2402.09353](https://arxiv.org/abs/2402.09353) | Décomposer le poids en magnitude et direction |
| **rsLoRA** | Kalajdzievski, *A Rank Stabilization Scaling Factor for Fine-Tuning with LoRA* (2023) | [2312.03732](https://arxiv.org/abs/2312.03732) | Échelle α/√r stable pour les rangs élevés |
| **PiSSA** | Meng et al., *PiSSA: Principal Singular Values and Singular Vectors Adaptation of Large Language Models* (2024) | [2404.02948](https://arxiv.org/abs/2404.02948) | Initialiser A et B par les valeurs singulières principales |
| **LoHa** | Nam Hyeon-Woo et al., *FedPara: Low-Rank Hadamard Product for Communication-Efficient Federated Learning* (2021) | [2108.06098](https://arxiv.org/abs/2108.06098) | Produit de Hadamard de matrices de rang faible |
| **LoKr (LyCORIS)** | Yeh et al., *Navigating Text-To-Image Customization: From LyCORIS Fine-Tuning to Model Evaluation* (2023) | [2309.14859](https://arxiv.org/abs/2309.14859) | Produit de Kronecker, famille LyCORIS pour la diffusion |
| **IA³** | Liu et al., *Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper than In-Context Learning* (2022) | [2205.05638](https://arxiv.org/abs/2205.05638) | Vecteurs qui multiplient les activations |
| **Prompt tuning** | Lester et al., *The Power of Scale for Parameter-Efficient Prompt Tuning* (2021) | [2104.08691](https://arxiv.org/abs/2104.08691) | Tokens virtuels appris en entrée |
| **Prefix tuning** | Li & Liang, *Prefix-Tuning: Optimizing Continuous Prompts for Generation* (2021) | [2101.00190](https://arxiv.org/abs/2101.00190) | Clés et valeurs virtuelles apprises dans chaque couche |
| **LoRA et oubli** | Biderman et al., *LoRA Learns Less and Forgets Less* (2024) | [2405.09673](https://arxiv.org/abs/2405.09673) | LoRA apprend moins mais oublie moins que le fine-tuning complet |
| **DreamBooth** | Ruiz et al., *DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation* (2022) | [2208.12242](https://arxiv.org/abs/2208.12242) | Apprendre un sujet à un modèle de diffusion avec quelques images |

## 04 — Optimisation des préférences

| Technique | Papier | Lien | Ce qu'il apporte |
|---|---|---|---|
| **DPO** | Rafailov et al., *Direct Preference Optimization: Your Language Model is Secretly a Reward Model* (2023) | [2305.18290](https://arxiv.org/abs/2305.18290) | Préférences optimisées directement, sans modèle de récompense ni RL |
| **IPO** | Azar et al., *A General Theoretical Paradigm to Understand Learning from Human Preferences* (2023) | [2310.12036](https://arxiv.org/abs/2310.12036) | Perte quadratique, moins de surapprentissage des préférences |
| **rDPO** | Chowdhury et al., *Provably Robust DPO: Aligning Language Models with Noisy Feedback* (2024) | [2403.00409](https://arxiv.org/abs/2403.00409) | DPO robuste aux étiquettes bruitées |
| **SLiC-HF** | Zhao et al., *SLiC-HF: Sequence Likelihood Calibration with Human Feedback* (2023) | [2305.10425](https://arxiv.org/abs/2305.10425) | Calibration de la vraisemblance par perte hinge |
| **APO** | D'Oosterlinck et al., *Anchored Preference Optimization and Contrastive Revisions: Addressing Underspecification in Alignment* (2024) | [2408.06266](https://arxiv.org/abs/2408.06266) | Préférences ancrées : monter la préférée ou baisser la rejetée |
| **NCA** | Chen et al., *Noise Contrastive Alignment of Language Models with Explicit Rewards* (2024) | [2402.05369](https://arxiv.org/abs/2402.05369) | Alignement par estimation contrastive du bruit |
| **BCO** | Jung et al., *Binary Classifier Optimization for Large Language Model Alignment* (2024) | [2404.04656](https://arxiv.org/abs/2404.04656) | Préférences vues comme un classifieur binaire |
| **SPPO** | Wu et al., *Self-Play Preference Optimization for Language Model Alignment* (2024) | [2405.00675](https://arxiv.org/abs/2405.00675) | Préférences comme un jeu à deux joueurs (self-play) |
| **EXO** | Ji et al., *Towards Efficient Exact Optimization of Language Model Alignment* (2024) | [2402.00856](https://arxiv.org/abs/2402.00856) | Optimisation exacte de l'objectif d'alignement |
| **AOT** | Melnyk et al., *Distributional Preference Alignment of LLMs via Optimal Transport* (2024) | [2406.05882](https://arxiv.org/abs/2406.05882) | Alignement de distributions par transport optimal |
| **DiscoPOP** | Lu et al., *Discovering Preference Optimization Algorithms with and for Large Language Models* (2024) | [2406.08414](https://arxiv.org/abs/2406.08414) | Perte de préférence découverte par un LLM |
| **ORPO** | Hong et al., *ORPO: Monolithic Preference Optimization without Reference Model* (2024) | [2403.07691](https://arxiv.org/abs/2403.07691) | SFT et préférences en une étape, sans référence |
| **SimPO** | Meng et al., *SimPO: Simple Preference Optimization with a Reference-Free Reward* (2024) | [2405.14734](https://arxiv.org/abs/2405.14734) | Récompense = log-probabilité moyenne, avec marge, sans référence |
| **CPO** | Xu et al., *Contrastive Preference Optimization: Pushing the Boundaries of LLM Performance in Machine Translation* (2024) | [2401.08417](https://arxiv.org/abs/2401.08417) | Préférences contrastives sans référence (traduction) |
| **KTO** | Ethayarajh et al., *KTO: Model Alignment as Prospect Theoretic Optimization* (2024) | [2402.01306](https://arxiv.org/abs/2402.01306) | Réponses non appariées : seulement bonne ou mauvaise |
| **RPO** | Pang et al., *Iterative Reasoning Preference Optimization* (2024) | [2404.19733](https://arxiv.org/abs/2404.19733) | DPO + terme de vraisemblance sur la réponse préférée |
| **Déplacement de vraisemblance** | Razin et al., *Unintentional Unalignment: Likelihood Displacement in Direct Preference Optimization* (2024) | [2410.08847](https://arxiv.org/abs/2410.08847) | Pourquoi DPO peut faire baisser la probabilité de la réponse préférée |
| **Self-Rewarding** | Yuan et al., *Self-Rewarding Language Models* (2024) | [2401.10020](https://arxiv.org/abs/2401.10020) | Le modèle juge ses propres réponses pour un DPO itératif |
| **DPO itératif** | Xiong et al., *Iterative Preference Learning from Human Feedback: Bridging Theory and Practice for RLHF under KL-Constraint* (2023) | [2312.11456](https://arxiv.org/abs/2312.11456) | Préférences on-policy répétées : théorie et pratique |
| **Tülu 3** | Lambert et al., *Tulu 3: Pushing Frontiers in Open Language Model Post-Training* (2024) | [2411.15124](https://arxiv.org/abs/2411.15124) | Recette complète ouverte : SFT, DPO, RLVR |

## 05 — RL en ligne

| Technique | Papier | Lien | Ce qu'il apporte |
|---|---|---|---|
| **PPO** | Schulman et al., *Proximal Policy Optimization Algorithms* (2017) | [1707.06347](https://arxiv.org/abs/1707.06347) | Policy gradient avec clipping et critique appris |
| **RLHF** | Stiennon et al., *Learning to summarize from human feedback* (2020) | [2009.01325](https://arxiv.org/abs/2009.01325) | Modèle de récompense appris sur des préférences humaines, puis PPO |
| **Constitutional AI** | Bai et al., *Constitutional AI: Harmlessness from AI Feedback* (2022) | [2212.08073](https://arxiv.org/abs/2212.08073) | Critique-révision et RL depuis un retour d'IA guidé par des principes |
| **RLAIF** | Lee et al., *RLAIF vs. RLHF: Scaling Reinforcement Learning from Human Feedback with AI Feedback* (2023) | [2309.00267](https://arxiv.org/abs/2309.00267) | Un LLM remplace les annotateurs humains |
| **GRPO** | Shao et al., *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models* (2024) | [2402.03300](https://arxiv.org/abs/2402.03300) | Avantages relatifs au groupe, sans critique |
| **DeepSeek-R1** | DeepSeek-AI, *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning* (2025) | [2501.12948](https://arxiv.org/abs/2501.12948) | RL à récompense vérifiable pour le raisonnement ; modèles R1-Distill |
| **Dr.GRPO** | Liu et al., *Understanding R1-Zero-Like Training: A Critical Perspective* (2025) | [2503.20783](https://arxiv.org/abs/2503.20783) | Retirer les biais de longueur et de difficulté de GRPO |
| **DAPO** | Yu et al., *DAPO: An Open-Source LLM Reinforcement Learning System at Scale* (2025) | [2503.14476](https://arxiv.org/abs/2503.14476) | Clipping asymétrique, normalisation par token, masque des réponses tronquées |
| **GSPO** | Zheng et al., *Group Sequence Policy Optimization* (2025) | [2507.18071](https://arxiv.org/abs/2507.18071) | Ratio d'importance au niveau de la séquence |
| **CISPO (MiniMax-M1)** | MiniMax, *MiniMax-M1: Scaling Test-Time Compute Efficiently with Lightning Attention* (2025) | [2506.13585](https://arxiv.org/abs/2506.13585) | Clipper les poids d'importance plutôt que les mises à jour |
| **RLOO** | Ahmadian et al., *Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback in LLMs* (2024) | [2402.14740](https://arxiv.org/abs/2402.14740) | REINFORCE avec baseline leave-one-out |

## 06 — Distillation

| Technique | Papier | Lien | Ce qu'il apporte |
|---|---|---|---|
| **Distillation de logits** | Hinton et al., *Distilling the Knowledge in a Neural Network* (2015) | [1503.02531](https://arxiv.org/abs/1503.02531) | Imiter les probabilités adoucies de l'enseignant |
| **Distillation de séquences** | Kim & Rush, *Sequence-Level Knowledge Distillation* (2016) | [1606.07947](https://arxiv.org/abs/1606.07947) | Entraîner l'élève sur les sorties générées par l'enseignant |
| **DistilBERT** | Sanh et al., *DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter* (2019) | [1910.01108](https://arxiv.org/abs/1910.01108) | BERT réduit de 40 % par distillation |
| **TinyBERT** | Jiao et al., *TinyBERT: Distilling BERT for Natural Language Understanding* (2019) | [1909.10351](https://arxiv.org/abs/1909.10351) | Distillation des couches intermédiaires, données augmentées |
| **DeiT** | Touvron et al., *Training data-efficient image transformers & distillation through attention* (2020) | [2012.12877](https://arxiv.org/abs/2012.12877) | Distillation d'un transformer de vision par token dédié |
| **GKD** | Agarwal et al., *On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes* (2023) | [2306.13649](https://arxiv.org/abs/2306.13649) | Distillation on-policy sur les propres générations de l'élève |
| **MiniLLM** | Gu et al., *MiniLLM: On-Policy Distillation of Large Language Models* (2023) | [2306.08543](https://arxiv.org/abs/2306.08543) | Distillation par KL inverse pour les LLM génératifs |
| **Minitron** | Muralidharan et al., *Compact Language Models via Pruning and Knowledge Distillation* (2024) | [2407.14679](https://arxiv.org/abs/2407.14679) | Élagage puis distillation pour produire des modèles plus petits |

## 07 — Données synthétiques

| Technique | Papier | Lien | Ce qu'il apporte |
|---|---|---|---|
| **STaR** | Zelikman et al., *STaR: Bootstrapping Reasoning With Reasoning* (2022) | [2203.14465](https://arxiv.org/abs/2203.14465) | Garder les raisonnements qui mènent à la bonne réponse, puis réentraîner |
| **RFT** | Yuan et al., *Scaling Relationship on Learning Mathematical Reasoning with Large Language Models* (2023) | [2308.01825](https://arxiv.org/abs/2308.01825) | Rejection sampling fine-tuning sur le raisonnement mathématique |
| **ReST-EM** | Singh et al., *Beyond Human Data: Scaling Self-Training for Problem-Solving with Language Models* (2023) | [2312.06585](https://arxiv.org/abs/2312.06585) | Boucle génération, filtrage, entraînement à grande échelle |
| **Self-Instruct** | Wang et al., *Self-Instruct: Aligning Language Models with Self-Generated Instructions* (2022) | [2212.10560](https://arxiv.org/abs/2212.10560) | Le modèle invente de nouvelles instructions à partir d'exemples |
| **Evol-Instruct (WizardLM)** | Xu et al., *WizardLM: Empowering large pre-trained language models to follow complex instructions* (2023) | [2304.12244](https://arxiv.org/abs/2304.12244) | Complexifier progressivement les instructions générées |
| **Model collapse** | Shumailov et al., *The Curse of Recursion: Training on Generated Data Makes Models Forget* (2023) | [2305.17493](https://arxiv.org/abs/2305.17493) | Entraîner sur ses propres sorties non filtrées dégrade le modèle |

## 08 — Fusion de modèles

| Technique | Papier | Lien | Ce qu'il apporte |
|---|---|---|---|
| **Model soups** | Wortsman et al., *Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time* (2022) | [2203.05482](https://arxiv.org/abs/2203.05482) | Moyenne des poids de plusieurs fine-tunes |
| **WiSE-FT** | Wortsman et al., *Robust fine-tuning of zero-shot models* (2021) | [2109.01903](https://arxiv.org/abs/2109.01903) | Interpoler fine-tune et modèle d'origine pour garder la robustesse |
| **Arithmétique des tâches** | Ilharco et al., *Editing Models with Task Arithmetic* (2022) | [2212.04089](https://arxiv.org/abs/2212.04089) | Additionner ou soustraire des vecteurs de tâche |
| **TIES-Merging** | Yadav et al., *TIES-Merging: Resolving Interference When Merging Models* (2023) | [2306.01708](https://arxiv.org/abs/2306.01708) | Élaguer, élire le signe, fusionner les coordonnées d'accord |
| **DARE** | Yu et al., *Language Models are Super Mario: Absorbing Abilities from Homologous Models as a Free Lunch* (2023) | [2311.03099](https://arxiv.org/abs/2311.03099) | Abandon aléatoire du vecteur de tâche et remise à l'échelle |
| **Fusion évolutionnaire** | Akiba et al., *Evolutionary Optimization of Model Merging Recipes* (2024) | [2403.13187](https://arxiv.org/abs/2403.13187) | Chercher la recette de fusion par algorithme évolutionnaire |
| **mergekit** | Goddard et al., *Arcee's MergeKit: A Toolkit for Merging Large Language Models* (2024) | [2403.13257](https://arxiv.org/abs/2403.13257) | Outil de fusion de grands modèles |
| **SLERP** | Shoemake, *Animating Rotation with Quaternion Curves* (SIGGRAPH 1985) | — | Interpolation sphérique, reprise pour les poids |

## 09 — Édition de connaissances

| Technique | Papier | Lien | Ce qu'il apporte |
|---|---|---|---|
| **ROME** | Meng et al., *Locating and Editing Factual Associations in GPT* (2022) | [2202.05262](https://arxiv.org/abs/2202.05262) | Localiser un fait dans un MLP et l'éditer par une mise à jour de rang 1 |
| **MEMIT** | Meng et al., *Mass-Editing Memory in a Transformer* (2022) | [2210.07229](https://arxiv.org/abs/2210.07229) | Éditer des milliers de faits répartis sur plusieurs couches |
| **AlphaEdit** | Fang et al., *AlphaEdit: Null-Space Constrained Knowledge Editing for Language Models* (2024) | [2410.02355](https://arxiv.org/abs/2410.02355) | Projeter la mise à jour hors de l'espace des connaissances à préserver |
| **MEND** | Mitchell et al., *Fast Model Editing at Scale* (2021) | [2110.11309](https://arxiv.org/abs/2110.11309) | Un réseau apprend à transformer le gradient en édition |
| **GRACE** | Hartvigsen et al., *Aging with GRACE: Lifelong Model Editing with Discrete Key-Value Adaptors* (2022) | [2211.11031](https://arxiv.org/abs/2211.11031) | Mémoire externe clé-valeur pour l'édition continue |
| **Effets de propagation** | Cohen et al., *Evaluating the Ripple Effects of Knowledge Editing in Language Models* (2023) | [2307.12976](https://arxiv.org/abs/2307.12976) | Mesurer si une édition se propage aux faits liés |
| **EasyEdit** | Wang et al., *EasyEdit: An Easy-to-use Knowledge Editing Framework for Large Language Models* (2023) | [2308.07269](https://arxiv.org/abs/2308.07269) | Cadre d'édition de connaissances |

## 10 — Désapprentissage

| Technique | Papier | Lien | Ce qu'il apporte |
|---|---|---|---|
| **TOFU** | Maini et al., *TOFU: A Task of Fictitious Unlearning for LLMs* (2024) | [2401.06121](https://arxiv.org/abs/2401.06121) | Benchmark : oublier des auteurs fictifs appris par fine-tuning |
| **NPO** | Zhang et al., *Negative Preference Optimization: From Catastrophic Collapse to Effective Unlearning* (2024) | [2404.05868](https://arxiv.org/abs/2404.05868) | Perte bornée de préférence négative, sans effondrement |
| **SimNPO** | Fan et al., *Simplicity Prevails: Rethinking Negative Preference Optimization for LLM Unlearning* (2024) | [2410.07163](https://arxiv.org/abs/2410.07163) | NPO sans modèle de référence |
| **Who's Harry Potter?** | Eldan & Russinovich, *Who's Harry Potter? Approximate Unlearning in LLMs* (2023) | [2310.02238](https://arxiv.org/abs/2310.02238) | Désapprentissage approché d'un corpus |
| **WMDP / RMU** | Li et al., *The WMDP Benchmark: Measuring and Reducing Malicious Use With Unlearning* (2024) | [2403.03218](https://arxiv.org/abs/2403.03218) | Benchmark de savoirs dangereux et méthode RMU |
| **MUSE** | Shi et al., *MUSE: Machine Unlearning Six-Way Evaluation for Language Models* (2024) | [2407.06460](https://arxiv.org/abs/2407.06460) | Évaluation du désapprentissage selon six critères |
| **ESD** | Gandikota et al., *Erasing Concepts from Diffusion Models* (2023) | [2303.07345](https://arxiv.org/abs/2303.07345) | Effacer un concept d'un modèle de diffusion |

## 11 — Pilotage par activations

| Technique | Papier | Lien | Ce qu'il apporte |
|---|---|---|---|
| **ActAdd** | Turner et al., *Steering Language Models With Activation Engineering* (2023) | [2308.10248](https://arxiv.org/abs/2308.10248) | Ajouter une différence d'activations pour orienter la génération |
| **CAA** | Panickssery et al., *Steering Llama 2 via Contrastive Activation Addition* (2023) | [2312.06681](https://arxiv.org/abs/2312.06681) | Vecteurs de pilotage moyennés sur des paires contrastées |
| **RepE** | Zou et al., *Representation Engineering: A Top-Down Approach to AI Transparency* (2023) | [2310.01405](https://arxiv.org/abs/2310.01405) | Lire et contrôler des concepts dans les représentations |
| **Direction du refus** | Arditi et al., *Refusal in Language Models Is Mediated by a Single Direction* (2024) | [2406.11717](https://arxiv.org/abs/2406.11717) | Un comportement porté par une seule direction, et son ablation |
| **Scaling Monosemanticity** | Templeton et al., *Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet* (2024) | [transformer-circuits.pub](https://transformer-circuits.pub/2024/scaling-monosemanticity/) | Features interprétables par autoencodeur creux, et pilotage par features |
| **Concept Sliders** | Gandikota et al., *Concept Sliders: LoRA Adaptors for Precise Control in Diffusion Models* (2023) | [2311.12092](https://arxiv.org/abs/2311.12092) | Curseurs de concepts pour les modèles de diffusion |

## 12 — Extension de contexte

| Technique | Papier | Lien | Ce qu'il apporte |
|---|---|---|---|
| **Position Interpolation** | Chen et al., *Extending Context Window of Large Language Models via Positional Interpolation* (2023) | [2306.15595](https://arxiv.org/abs/2306.15595) | Compresser les positions pour rester dans la plage vue à l'entraînement |
| **NTK-aware RoPE** | bloc97, *NTK-Aware Scaled RoPE* (post r/LocalLLaMA, 2023) | formalisé dans YaRN | Augmenter la base de RoPE plutôt que compresser les positions |
| **YaRN** | Peng et al., *YaRN: Efficient Context Window Extension of Large Language Models* (2023) | [2309.00071](https://arxiv.org/abs/2309.00071) | Interpolation par bande de fréquence + correction de température |
| **ABF** | Xiong et al., *Effective Long-Context Scaling of Foundation Models* (2023) | [2309.16039](https://arxiv.org/abs/2309.16039) | Augmenter la base de RoPE et pré-entraîner sur du long |
| **Code Llama** | Rozière et al., *Code Llama: Open Foundation Models for Code* (2023) | [2308.12950](https://arxiv.org/abs/2308.12950) | Fine-tuning long contexte avec base RoPE augmentée |
| **LongRoPE** | Ding et al., *LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens* (2024) | [2402.13753](https://arxiv.org/abs/2402.13753) | Facteurs de mise à l'échelle par dimension trouvés par recherche |
| **Ring Attention** | Liu et al., *Ring Attention with Blockwise Transformers for Near-Infinite Context* (2023) | [2310.01889](https://arxiv.org/abs/2310.01889) | Répartir une séquence très longue entre plusieurs GPU |
| **RULER** | Hsieh et al., *RULER: What's the Real Context Size of Your Long-Context Language Models?* (2024) | [2404.06654](https://arxiv.org/abs/2404.06654) | Mesurer la taille de contexte réellement utilisable |

## 13 — Extension de vocabulaire

| Technique | Papier | Lien | Ce qu'il apporte |
|---|---|---|---|
| **Init par moyenne** | Hewitt, *Initializing New Word Embeddings for Pretrained Language Models* (2021) | [billet](https://cs.columbia.edu/~johnhew/vocab-expansion.html) | Initialiser les nouveaux embeddings à la moyenne des existants |
| **Chinese-LLaMA** | Cui et al., *Efficient and Effective Text Encoding for Chinese LLaMA and Alpaca* (2023) | [2304.08177](https://arxiv.org/abs/2304.08177) | Étendre le vocabulaire de LLaMA au chinois, puis pré-entraîner |
| **Swallow** | Fujii et al., *Continual Pre-Training for Cross-Lingual LLM Adaptation: Enhancing Japanese Language Capabilities* (2024) | [2404.17790](https://arxiv.org/abs/2404.17790) | Adapter un LLM au japonais : vocabulaire et pré-entraînement continu |
| **WECHSEL** | Minixhofer et al., *WECHSEL: Effective initialization of subword embeddings for cross-lingual transfer of monolingual language models* (2021) | [2112.06598](https://arxiv.org/abs/2112.06598) | Initialiser les embeddings d'une nouvelle langue par alignement multilingue |
| **FOCUS** | Dobler & Melo, *FOCUS: Effective Embedding Initialization for Monolingual Specialization of Multilingual Models* (2023) | [2305.14481](https://arxiv.org/abs/2305.14481) | Initialiser les nouveaux tokens à partir de tokens similaires |
| **ZeTT** | Minixhofer et al., *Zero-Shot Tokenizer Transfer* (2024) | [2405.07883](https://arxiv.org/abs/2405.07883) | Hyper-réseau qui prédit les embeddings d'un nouveau tokenizer |

## 14 — Croissance d'architecture

| Technique | Papier | Lien | Ce qu'il apporte |
|---|---|---|---|
| **Empilement progressif** | Gong et al., *Efficient Training of BERT by Progressively Stacking* (ICML 2019) | [PMLR](https://proceedings.mlr.press/v97/gong19a.html) | Doubler la profondeur en copiant les couches entraînées |
| **SOLAR (depth up-scaling)** | Kim et al., *SOLAR 10.7B: Scaling Large Language Models with Simple yet Effective Depth Up-Scaling* (2023) | [2312.15166](https://arxiv.org/abs/2312.15166) | Dupliquer et coudre des couches, puis poursuivre l'entraînement |
| **LLaMA Pro** | Wu et al., *LLaMA Pro: Progressive LLaMA with Block Expansion* (2024) | [2401.02415](https://arxiv.org/abs/2401.02415) | Insérer des blocs identité, seuls entraînés |
| **Sparse upcycling** | Komatsuzaki et al., *Sparse Upcycling: Training Mixture-of-Experts from Dense Checkpoints* (2022) | [2212.05055](https://arxiv.org/abs/2212.05055) | Transformer un modèle dense en MoE en copiant les MLP |
| **Mixtral** | Jiang et al., *Mixtral of Experts* (2024) | [2401.04088](https://arxiv.org/abs/2401.04088) | Architecture MoE cible (8 experts, top-2) |
| **Upcycling de LLM** | He et al., *Upcycling Large Language Models into Mixture of Experts* (2024) | [2410.07524](https://arxiv.org/abs/2410.07524) | Recette d'upcycling à grande échelle |
| **V-MoE** | Riquelme et al., *Scaling Vision with Sparse Mixture of Experts* (2021) | [2106.05974](https://arxiv.org/abs/2106.05974) | Mixture d'experts pour la vision |
| **MoE-LLaVA** | Lin et al., *MoE-LLaVA: Mixture of Experts for Large Vision-Language Models* (2024) | [2401.15947](https://arxiv.org/abs/2401.15947) | Mixture d'experts pour les modèles vision-langage |

## 15 — Quantification

| Technique | Papier | Lien | Ce qu'il apporte |
|---|---|---|---|
| **LLM.int8()** | Dettmers et al., *LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale* (2022) | [2208.07339](https://arxiv.org/abs/2208.07339) | Quantification int8 avec traitement des valeurs aberrantes |
| **NF4 (QLoRA)** | Dettmers et al., *QLoRA: Efficient Finetuning of Quantized LLMs* (2023) | [2305.14314](https://arxiv.org/abs/2305.14314) | Format 4 bits adapté aux poids de distribution normale |
| **GPTQ** | Frantar et al., *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers* (2022) | [2210.17323](https://arxiv.org/abs/2210.17323) | Quantification calibrée corrigeant les erreurs colonne par colonne |
| **AWQ** | Lin et al., *AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration* (2023) | [2306.00978](https://arxiv.org/abs/2306.00978) | Protéger les canaux importants par mise à l'échelle |
| **SmoothQuant** | Xiao et al., *SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models* (2022) | [2211.10438](https://arxiv.org/abs/2211.10438) | Déplacer la difficulté des activations vers les poids (W8A8) |
| **FP8** | Micikevicius et al., *FP8 Formats for Deep Learning* (2022) | [2209.05433](https://arxiv.org/abs/2209.05433) | Formats flottants 8 bits pour l'entraînement et l'inférence |
| **LLM-QAT** | Liu et al., *LLM-QAT: Data-Free Quantization Aware Training for Large Language Models* (2023) | [2305.17888](https://arxiv.org/abs/2305.17888) | Quantification pendant l'entraînement pour les LLM |
| **torchao QAT** | PyTorch, *Quantization-Aware Training for Large Language Models with PyTorch* (2024) | [billet](https://pytorch.org/blog/quantization-aware-training/) | Recette QAT de torchao utilisée par l'exemple |
| **k-quants** | Kawrakow, *k-quants* (llama.cpp, 2023) | [PR 1684](https://github.com/ggml-org/llama.cpp/pull/1684) | Quantification par blocs et super-blocs du format GGUF |

## 16 — Élagage

| Technique | Papier | Lien | Ce qu'il apporte |
|---|---|---|---|
| **Élagage par magnitude** | Han et al., *Learning both Weights and Connections for Efficient Neural Networks* (2015) | [1506.02626](https://arxiv.org/abs/1506.02626) | Retirer les poids de faible valeur absolue, puis réentraîner |
| **SparseGPT** | Frantar & Alistarh, *SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot* (2023) | [2301.00774](https://arxiv.org/abs/2301.00774) | Élagage en une passe avec reconstruction des poids |
| **Wanda** | Sun et al., *A Simple and Effective Pruning Approach for Large Language Models* (2023) | [2306.11695](https://arxiv.org/abs/2306.11695) | Score |poids| × norme de l'activation, sans mise à jour |
| **Sparsité 2:4** | Mishra et al., *Accelerating Sparse Deep Neural Networks* (2021) | [2104.08378](https://arxiv.org/abs/2104.08378) | Motif 2 sur 4 accéléré par le matériel |
| **ShortGPT** | Men et al., *ShortGPT: Layers in Large Language Models are More Redundant Than You Expect* (2024) | [2403.03853](https://arxiv.org/abs/2403.03853) | Retirer les couches à faible influence sur le flux résiduel |
| **Couches profondes** | Gromov et al., *The Unreasonable Ineffectiveness of the Deeper Layers* (2024) | [2403.17887](https://arxiv.org/abs/2403.17887) | Beaucoup de couches profondes se retirent avec peu de perte |
| **LLM-Pruner** | Ma et al., *LLM-Pruner: On the Structural Pruning of Large Language Models* (2023) | [2305.11627](https://arxiv.org/abs/2305.11627) | Élagage structuré de LLM et récupération par LoRA |
| **Minitron** | Muralidharan et al., *Compact Language Models via Pruning and Knowledge Distillation* (2024) | [2407.14679](https://arxiv.org/abs/2407.14679) | Élagage en largeur et profondeur puis distillation |
| **BK-SDM** | Kim et al., *BK-SDM: A Lightweight, Fast, and Cheap Version of Stable Diffusion* (2023) | [2305.15798](https://arxiv.org/abs/2305.15798) | Retirer des blocs d'un UNet de diffusion puis distiller |

## 17 — Optimisation du prompt

| Technique | Papier | Lien | Ce qu'il apporte |
|---|---|---|---|
| **Few-shot (GPT-3)** | Brown et al., *Language Models are Few-Shot Learners* (2020) | [2005.14165](https://arxiv.org/abs/2005.14165) | Spécifier une tâche par quelques exemples dans le prompt |
| **Chain-of-thought** | Wei et al., *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models* (2022) | [2201.11903](https://arxiv.org/abs/2201.11903) | Faire raisonner étape par étape par l'exemple |
| **Zero-shot CoT** | Kojima et al., *Large Language Models are Zero-Shot Reasoners* (2022) | [2205.11916](https://arxiv.org/abs/2205.11916) | « Let's think step by step » sans exemple |
| **KATE** | Liu et al., *What Makes Good In-Context Examples for GPT-3?* (2021) | [2101.06804](https://arxiv.org/abs/2101.06804) | Choisir les exemples les plus proches de la question |
| **APE** | Zhou et al., *Large Language Models Are Human-Level Prompt Engineers* (2022) | [2211.01910](https://arxiv.org/abs/2211.01910) | Générer et sélectionner automatiquement des instructions |
| **OPRO** | Yang et al., *Large Language Models as Optimizers* (2023) | [2309.03409](https://arxiv.org/abs/2309.03409) | Un LLM optimiseur propose des instructions à partir des scores passés |
| **DSPy** | Khattab et al., *DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines* (2023) | [2310.03714](https://arxiv.org/abs/2310.03714) | Programmes LLM déclaratifs optimisés automatiquement |
| **MIPROv2** | Opsahl-Ong et al., *Optimizing Instructions and Demonstrations for Multi-Stage Language Model Programs* (2024) | [2406.11695](https://arxiv.org/abs/2406.11695) | Optimisation conjointe des instructions et des exemples |

## 18 — RAG

| Technique | Papier | Lien | Ce qu'il apporte |
|---|---|---|---|
| **RAG** | Lewis et al., *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks* (2020) | [2005.11401](https://arxiv.org/abs/2005.11401) | Récupérer des passages et générer à partir d'eux |
| **BM25** | Robertson & Zaragoza, *The Probabilistic Relevance Framework: BM25 and Beyond* (2009) | [DOI](https://doi.org/10.1561/1500000019) | Recherche lexicale pondérée par la rareté des mots |
| **DPR** | Karpukhin et al., *Dense Passage Retrieval for Open-Domain Question Answering* (2020) | [2004.04906](https://arxiv.org/abs/2004.04906) | Recherche dense par embeddings de question et de passage |
| **Sentence-BERT** | Reimers & Gurevych, *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks* (2019) | [1908.10084](https://arxiv.org/abs/1908.10084) | Embeddings de phrases pour la similarité |
| **BGE-M3** | Chen et al., *M3-Embedding: Multi-Linguality, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation* (2024) | [2402.03216](https://arxiv.org/abs/2402.03216) | Embeddings multilingues dense, creux et multi-vecteurs |
| **ColBERT** | Khattab & Zaharia, *ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT* (2020) | [2004.12832](https://arxiv.org/abs/2004.12832) | Recherche multi-vecteurs par interaction tardive |
| **RRF** | Cormack, Clarke & Buettcher, *Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods* (SIGIR 2009) | [DOI](https://doi.org/10.1145/1571941.1572114) | Fusionner des classements par leurs rangs |
| **Reranking BERT** | Nogueira & Cho, *Passage Re-ranking with BERT* (2019) | [1901.04085](https://arxiv.org/abs/1901.04085) | Cross-encoder qui reclasse les candidats |
| **Lost in the Middle** | Liu et al., *Lost in the Middle: How Language Models Use Long Contexts* (2023) | [2307.03172](https://arxiv.org/abs/2307.03172) | Les passages au milieu d'un long contexte sont moins utilisés |
| **Panorama RAG** | Gao et al., *Retrieval-Augmented Generation for Large Language Models: A Survey* (2023) | [2312.10997](https://arxiv.org/abs/2312.10997) | Synthèse des architectures RAG |

## 19 — Outils et agents

| Technique | Papier | Lien | Ce qu'il apporte |
|---|---|---|---|
| **ReAct** | Yao et al., *ReAct: Synergizing Reasoning and Acting in Language Models* (2022) | [2210.03629](https://arxiv.org/abs/2210.03629) | Alterner raisonnement et actions dans une boucle |
| **Toolformer** | Schick et al., *Toolformer: Language Models Can Teach Themselves to Use Tools* (2023) | [2302.04761](https://arxiv.org/abs/2302.04761) | Le modèle apprend seul quand appeler une API |
| **Gorilla** | Patil et al., *Gorilla: Large Language Model Connected with Massive APIs* (2023) | [2305.15334](https://arxiv.org/abs/2305.15334) | LLM entraîné à appeler correctement de nombreuses API |

## 20 — Calcul à l'inférence

| Technique | Papier | Lien | Ce qu'il apporte |
|---|---|---|---|
| **Self-consistency** | Wang et al., *Self-Consistency Improves Chain of Thought Reasoning in Language Models* (2022) | [2203.11171](https://arxiv.org/abs/2203.11171) | Vote majoritaire sur plusieurs raisonnements échantillonnés |
| **Vérificateurs (GSM8K)** | Cobbe et al., *Training Verifiers to Solve Math Word Problems* (2021) | [2110.14168](https://arxiv.org/abs/2110.14168) | Best-of-N par vérificateur appris ; jeu GSM8K |
| **PRM** | Lightman et al., *Let's Verify Step by Step* (2023) | [2305.20050](https://arxiv.org/abs/2305.20050) | Noter chaque étape du raisonnement plutôt que la réponse finale |
| **Self-Refine** | Madaan et al., *Self-Refine: Iterative Refinement with Self-Feedback* (2023) | [2303.17651](https://arxiv.org/abs/2303.17651) | Le modèle critique et corrige sa propre réponse |
| **Large Language Monkeys** | Brown et al., *Large Language Monkeys: Scaling Inference Compute with Repeated Sampling* (2024) | [2407.21787](https://arxiv.org/abs/2407.21787) | La couverture (pass@N) croît avec le nombre d'échantillons |
| **Calcul à l'inférence optimal** | Snell et al., *Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters* (2024) | [2408.03314](https://arxiv.org/abs/2408.03314) | Allouer le calcul d'inférence selon la difficulté |
