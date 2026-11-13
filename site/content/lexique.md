## ABF

Adjusted Base Frequency : augmente la base de fréquence de l'encodage RoPE pour qu'un modèle accepte des textes plus longs. Voir [Extension de contexte](#extension-contexte).

## accelerate

Bibliothèque Hugging Face qui lance un même script sur un ou plusieurs GPU, avec FSDP ou DeepSpeed. Voir [Installer et lancer](#installer).

## Adam, AdamW

Optimiseurs qui ajustent le pas d'apprentissage de chaque paramètre ; AdamW y ajoute une décroissance des poids. Il garde deux valeurs par paramètre entraîné, ce qui pèse lourd en mémoire. Voir [Fine-tuning supervisé](#fine-tuning-supervise).

## Adaptateur

Petit module entraînable ajouté à un modèle gelé ; on l'entraîne, le déploie et le retire sans toucher aux poids d'origine. Voir [PEFT](#peft).

## Agent

Programme dans lequel un modèle décide, tour après tour, quel outil appeler et quand s'arrêter, en voyant les résultats des appels précédents. Voir [Outils et agents](#outils-agents).

## AlphaEdit

Méthode d'édition de connaissances qui projette chaque modification hors des directions utilisées par les autres faits, pour les préserver. Voir [Édition de connaissances](#edition-connaissances).

## API compatible OpenAI

Interface HTTP standard (`/v1/chat/completions`) proposée par Ollama, vLLM et la plupart des fournisseurs ; les exemples 17 à 20 l'utilisent pour interroger un modèle servi ailleurs. Voir [Installer et lancer](#installer).

## Attention

Opération centrale du transformer : chaque token pondère les autres tokens du contexte selon leur pertinence. Son coût croît avec le carré de la longueur du texte. Voir [Extension de contexte](#extension-contexte).

## AWQ

Activation-aware Weight Quantization : quantification calibrée qui protège les canaux de poids les plus utiles en les mettant à l'échelle avant l'arrondi. Voir [Quantification](#quantification).

## Batch

Lot d'exemples traités ensemble à chaque pas d'entraînement ou de génération. L'accumulation de gradients simule un lot plus grand que la mémoire ne le permet. Voir [Fine-tuning supervisé](#fine-tuning-supervise).

## Best-of-N

Générer N réponses et garder celle qu'un juge (modèle de récompense, vérificateur, vraisemblance) note le mieux. Voir [Calcul à l'inférence](#calcul-inference).

## bf16, fp16, fp32

Formats de nombres à virgule flottante sur 16 ou 32 bits. Le bf16 garde la plage de valeurs du fp32 avec moins de précision : c'est le format courant des poids et des calculs. Voir [Quantification](#quantification).

## bitsandbytes

Bibliothèque de quantification à la volée en 8 et 4 bits (NF4), utilisée par QLoRA et transformers. Voir [Quantification](#quantification).

## BM25

Score de recherche lexicale qui pondère les mots de la requête par leur rareté dans le corpus. Brique de base du [RAG](#rag).

## BPE

Byte Pair Encoding : méthode de construction d'un tokenizer qui fusionne pas à pas les paires de caractères les plus fréquentes. Voir [Extension de vocabulaire](#extension-vocabulaire).

## Cache KV

Mémoire des clés et valeurs d'attention déjà calculées : en génération, chaque nouveau token réutilise celles des tokens précédents au lieu de tout recalculer. Voir [Calcul à l'inférence](#calcul-inference).

## Catastrophic forgetting

Voir [Oubli catastrophique](#lexterm-oubli-catastrophique).

## Chain-of-thought, CoT

Faire écrire au modèle son raisonnement pas à pas avant la réponse finale, ce qui améliore les tâches de calcul et de logique. Voir [Optimisation du prompt](#optimisation-prompt).

## Chat template

Format qui transforme une conversation (messages système, utilisateur, assistant) en texte attendu par un modèle instruct. Un modèle affiné doit garder le même. Voir [Fine-tuning supervisé](#fine-tuning-supervise).

## Cicatrisation

Court entraînement qui répare un modèle après une modification brutale (élagage, ajout de couches), souvent par distillation depuis le modèle d'origine. Voir [Élagage](#elagage).

## CPU

Processeur central de l'ordinateur. Il suffit pour la fusion de modèles, BM25 ou un modèle GGUF, mais reste lent pour entraîner un réseau. Voir [Fusion de modèles](#fusion-modeles).

## Cross-encoder

Modèle qui lit ensemble une question et un passage pour noter leur pertinence : plus précis qu'une comparaison d'embeddings, mais trop lent pour tout un corpus. Voir [RAG](#rag).

## CUDA

Plateforme de calcul des GPU NVIDIA. PyTorch y envoie des « noyaux », petits programmes qui font les multiplications de matrices sur le GPU. Voir [PEFT, exemple du dépôt](#peft-exemple).

## DAPO

Variante de GRPO (RL sur des groupes de réponses) qui découple les bornes de clipping, filtre les groupes sans signal et normalise la perte par token. Voir [RL en ligne](#rl-en-ligne).

## DAPT

Domain-Adaptive Pre-Training : pré-entraînement continu sur les textes d'un domaine (médical, juridique, code...). Voir [Pré-entraînement continu](#pre-entrainement-continu).

## DARE

Drop And REscale : avant de fusionner, met au hasard à zéro une partie de chaque vecteur de tâche et rééchelonne le reste. Combiné à TIES dans DARE-TIES. Voir [Fusion de modèles](#fusion-modeles).

## Désapprentissage

Faire oublier à un modèle un savoir précis (donnée personnelle, contenu protégé) en préservant le reste. Voir [Désapprentissage](#desapprentissage).

## Distillation

Entraîner un modèle élève à imiter un modèle enseignant, souvent plus grand, à partir de ses réponses ou de ses distributions de probabilités. Voir [Distillation](#distillation).

## DoRA

Weight-Decomposed Low-Rank Adaptation : variante de LoRA qui sépare la norme et la direction des poids et n'applique LoRA qu'à la direction. Voir [PEFT](#peft).

## DPO

Direct Preference Optimization : apprend à partir de paires « réponse préférée / réponse rejetée » par une simple perte de classification, sans modèle de récompense ni boucle de RL. Voir [Optimisation des préférences](#optimisation-preferences).

## DSPy

Bibliothèque qui écrit et optimise automatiquement les prompts et exemples d'un programme à base de LLM (optimiseur MIPROv2). Voir [Optimisation du prompt](#optimisation-prompt).

## Édition de connaissances

Modifier directement l'endroit d'un modèle qui stocke un fait, sans réentraînement. Voir [Édition de connaissances](#edition-connaissances).

## Élagage

Retirer d'un modèle des poids, des neurones ou des couches peu utiles pour le rendre plus petit ou plus rapide. Voir [Élagage](#elagage).

## Embedding

Vecteur qui représente un token ou un texte ; deux textes proches par le sens ont des vecteurs proches. Sert à la recherche dense du [RAG](#rag) et à l'entrée de tout modèle.

## Encodeur

Modèle qui lit un texte entier pour le représenter ou le classer (BERT, ModernBERT), par opposition au décodeur qui génère token par token. Voir [Fine-tuning supervisé](#fine-tuning-supervise).

## Époque

Un passage complet sur les données d'entraînement. Voir [Fine-tuning supervisé](#fine-tuning-supervise).

## Exactitude

Part des réponses justes sur un jeu de test. Les exemples du dépôt l'affichent avant et après la méthode. Voir [Statut des exemples](#statut-des-exemples).

## Few-shot, zero-shot

Donner quelques exemples résolus dans le prompt (few-shot), ou aucun (zero-shot). Voir [Optimisation du prompt](#optimisation-prompt).

## Fine-tuning

Poursuivre l'entraînement d'un modèle existant sur des données choisies pour l'adapter à une tâche. Complet quand tous les poids bougent, efficace (PEFT) quand seul un petit module apprend. Voir [Fine-tuning supervisé](#fine-tuning-supervise).

## Forward, backward

Passe avant (calcul de la sortie à partir de l'entrée) et passe arrière (calcul des gradients de la perte par rapport aux poids). Un pas d'entraînement fait les deux. Voir [PEFT, exemple du dépôt](#peft-exemple).

## FP8

Format flottant sur 8 bits, accéléré par les GPU récents, utilisé pour quantifier poids et activations. Voir [Quantification](#quantification).

## FSDP

Fully Sharded Data Parallel : répartit poids, gradients et état de l'optimiseur entre plusieurs GPU pour entraîner des modèles trop grands pour un seul. Voir [PEFT](#peft).

## Fusion de modèles

Combiner les poids de plusieurs modèles affinés depuis la même base, sans réentraînement. Voir [Fusion de modèles](#fusion-modeles).

## Function calling

Capacité d'un modèle à produire un appel d'outil structuré (nom de fonction et arguments) au lieu d'une réponse en texte. Voir [Outils et agents](#outils-agents).

## GGUF

Format de fichier de llama.cpp et d'Ollama pour les modèles quantifiés, exécutables sur CPU ou GPU. Voir [Quantification](#quantification).

## GKD

Generalized Knowledge Distillation : distillation sur les réponses générées par l'élève lui-même, notées par l'enseignant, pour réduire l'écart entre entraînement et génération. Voir [Distillation](#distillation).

## GPTQ

Quantification calibrée qui corrige les erreurs d'arrondi colonne par colonne à partir de quelques centaines de textes. Voir [Quantification](#quantification).

## GPU

Processeur graphique, fait pour les calculs massivement parallèles : c'est lui qui exécute les modèles. Sa mémoire (12 Go sur le poste de référence) limite la taille des modèles utilisables. Voir [PEFT, exemple du dépôt](#peft-exemple).

## Gradient checkpointing

Ne garde qu'une partie des activations pendant la passe avant et recalcule les autres pendant la passe arrière : beaucoup moins de mémoire, un peu plus de temps de calcul. Voir [Fine-tuning supervisé](#fine-tuning-supervise).

## GRPO

Group Relative Policy Optimization : RL qui fait générer plusieurs réponses par question et compare chacune à la moyenne de son groupe, sans modèle de valeur. Voir [RL en ligne](#rl-en-ligne).

## GSM8K

Jeu d'environ 8 800 problèmes de mathématiques d'école (7 473 d'entraînement, 1 319 de test), à réponse numérique vérifiable, référence pour le raisonnement. Voir [RL en ligne](#rl-en-ligne).

## Hook

Fonction branchée sur une couche d'un modèle PyTorch, appelée à chaque passe avant : elle peut lire ou modifier les activations. Voir [Pilotage par activations](#pilotage-activations).

## Hugging Face Hub

Plateforme qui héberge modèles et jeux de données ; transformers y télécharge les poids au premier lancement, puis les garde en cache. Voir [PEFT, exemple du dépôt](#peft-exemple).

## IA³

Infused Adapter by Inhibiting and Amplifying Inner Activations : méthode PEFT qui apprend des vecteurs multipliant les activations, avec environ 0,01 % des paramètres. Voir [PEFT](#peft).

## Inférence

Utilisation d'un modèle pour produire une réponse, par opposition à son entraînement. Voir [Calcul à l'inférence](#calcul-inference).

## Instruct (modèle)

Modèle de base affiné pour suivre des consignes et converser (SFT puis préférences), par opposition au modèle de base qui ne fait que continuer du texte. Voir [Fine-tuning supervisé](#fine-tuning-supervise).

## Intervalle de confiance

Plage dans laquelle se trouve probablement la vraie valeur d'une mesure ; un gain n'est démontré que s'il dépasse cette incertitude. Voir [Statut des exemples](#statut-des-exemples).

## KL (divergence)

Mesure de l'écart entre deux distributions de probabilités. Sert à la distillation et à retenir un modèle près de sa référence en RL et en DPO. Voir [Distillation](#distillation).

## KTO

Kahneman-Tversky Optimization : apprentissage de préférences à partir de réponses simplement jugées bonnes ou mauvaises, sans paires. Voir [Optimisation des préférences](#optimisation-preferences).

## llama.cpp

Moteur d'inférence en C++ pour les modèles GGUF, sur CPU ou GPU ; Ollama s'appuie dessus. Voir [Quantification](#quantification).

## LLM

Large Language Model : grand modèle de langage, en général un transformer décodeur qui prédit le token suivant. Voir [Quelle méthode pour quel besoin](#choisir-une-methode).

## Logits

Scores bruts produits par le modèle pour chaque token du vocabulaire, avant leur transformation en probabilités. La distillation par logits aligne ceux de l'élève sur ceux de l'enseignant. Voir [Distillation](#distillation).

## LoRA

Low-Rank Adaptation : ajoute à chaque couche linéaire deux petites matrices de rang faible, seules entraînées. Voir [PEFT](#peft).

## MEMIT

Mass-Editing Memory in a Transformer : édition de milliers de faits à la fois, répartie sur plusieurs couches. Voir [Édition de connaissances](#edition-connaissances).

## Mergekit

Outil de référence pour fusionner des modèles (linéaire, SLERP, TIES, DARE...) à partir d'un fichier de configuration. Voir [Fusion de modèles](#fusion-modeles).

## MLM

Masked Language Modeling : objectif des encodeurs comme BERT, qui devinent des mots masqués dans un texte. Voir [Pré-entraînement continu](#pre-entrainement-continu).

## MLP

Partie de chaque bloc du transformer qui transforme chaque token indépendamment ; on lui attribue le stockage de nombreux faits. Voir [Édition de connaissances](#edition-connaissances).

## Modèle de base

Modèle issu du seul pré-entraînement, qui continue du texte sans suivre de consignes. Point de départ commun des fine-tunes qu'on peut fusionner. Voir [Fusion de modèles](#fusion-modeles).

## Modèle de récompense

Modèle qui note une réponse ; entraîné sur des préférences humaines, il guide le RL ou départage les réponses en best-of-N. Voir [RL en ligne](#rl-en-ligne).

## MoE

Mixture of Experts : couches où un routeur envoie chaque token vers quelques sous-réseaux (experts) parmi beaucoup, pour plus de capacité à coût de calcul égal. Voir [Croissance d'architecture](#croissance-architecture).

## NF4

Format 4 bits dont les niveaux suivent une distribution normale, celle des poids d'un réseau. Utilisé par QLoRA. Voir [Quantification](#quantification).

## NPO

Negative Preference Optimization : désapprentissage qui traite les réponses à oublier comme des réponses rejetées, plus stable que la simple montée de gradient. Voir [Désapprentissage](#desapprentissage).

## NTK-aware

Mise à l'échelle de l'encodage RoPE qui étire surtout les basses fréquences, pour allonger le contexte sans réentraînement. Voir [Extension de contexte](#extension-contexte).

## Ollama

Serveur local de modèles GGUF, simple à installer, qui expose une API compatible OpenAI. Voir [Installer et lancer](#installer).

## On-policy, off-policy

Apprentissage sur des réponses produites par le modèle en cours d'entraînement (on-policy), ou sur des réponses écrites à l'avance (off-policy). Voir [Optimisation des préférences](#optimisation-preferences).

## OPRO

Optimization by PROmpting : un LLM propose de nouvelles consignes à partir des scores des précédentes, en boucle. Voir [Optimisation du prompt](#optimisation-prompt).

## ORPO

Odds Ratio Preference Optimization : combine SFT et préférences en une seule étape, sans modèle de référence. Voir [Optimisation des préférences](#optimisation-preferences).

## Oubli catastrophique

Perte des compétences générales d'un modèle quand on l'entraîne sur une tâche ou un domaine étroit. On le limite en remélangeant des données générales, avec LoRA, ou en restant près de la référence. Voir [Pré-entraînement continu](#pre-entrainement-continu).

## Passkey

Test d'extension de contexte : un code caché au milieu d'un long texte, que le modèle doit retrouver. Voir [Extension de contexte](#extension-contexte).

## PEFT

Parameter-Efficient Fine-Tuning : méthodes qui n'entraînent qu'une petite partie des paramètres (LoRA, IA³, prompt tuning...), et bibliothèque Hugging Face du même nom. Voir [PEFT](#peft).

## Perplexité

Mesure de la surprise d'un modèle face à un texte : plus elle est basse, mieux il le prédit. Sert à détecter l'oubli sur des textes généraux. Voir [Pré-entraînement continu](#pre-entrainement-continu).

## Pilotage par activations

Ajouter une direction aux activations internes pendant la génération pour orienter un trait (ton, sentiment) sans toucher aux poids. Voir [Pilotage par activations](#pilotage-activations).

## PiSSA

Variante de LoRA initialisée sur les valeurs singulières principales des poids, qui converge plus vite. Voir [PEFT](#peft).

## PPO

Proximal Policy Optimization : algorithme de RL historique du RLHF, avec un modèle de valeur et un clipping des mises à jour. Voir [RL en ligne](#rl-en-ligne).

## Pré-entraînement continu

Prolonger le pré-entraînement d'un modèle (prédiction du token suivant) sur un nouveau corpus de texte brut. Voir [Pré-entraînement continu](#pre-entrainement-continu).

## Prefix tuning, prompt tuning

Méthodes PEFT qui apprennent des tokens virtuels : ajoutés en entrée (prompt tuning) ou injectés dans chaque couche d'attention (prefix tuning). Voir [PEFT](#peft).

## PRM

Process Reward Model : modèle de récompense qui note chaque étape d'un raisonnement, pas seulement la réponse finale. Voir [Calcul à l'inférence](#calcul-inference).

## PTQ, QAT

Quantification après entraînement (Post-Training Quantization), ou pendant un court entraînement où le modèle apprend à tolérer l'arrondi (Quantization-Aware Training). Voir [Quantification](#quantification).

## QLoRA

LoRA entraînée sur un modèle chargé en 4 bits (NF4) : divise la mémoire par quatre environ. Voir [PEFT](#peft).

## Quantification

Stocker les poids (et parfois les activations) sur 8, 4 ou 3 bits au lieu de 16. Voir [Quantification](#quantification).

## RAG

Retrieval-Augmented Generation : chercher des passages pertinents et les placer dans le prompt au moment de répondre. Voir [RAG](#rag).

## ReAct

Schéma d'agent qui alterne raisonnement écrit et appel d'outil, chaque résultat nourrissant l'étape suivante. Voir [Outils et agents](#outils-agents).

## Rejection sampling

Générer plusieurs réponses, ne garder que celles qu'un vérificateur juge correctes, et entraîner le modèle dessus (RFT, STaR). Voir [Données synthétiques](#donnees-synthetiques).

## Replay

Remélanger des données générales aux données d'un nouvel entraînement pour limiter l'oubli catastrophique. Voir [Pré-entraînement continu](#pre-entrainement-continu).

## Reranker

Modèle (cross-encoder) qui relit chaque couple question-passage pour reclasser les candidats d'une recherche. Voir [RAG](#rag).

## RFT

Rejection sampling Fine-Tuning : entraînement sur les réponses du modèle filtrées par un vérificateur. Désigne aussi le Reinforcement Fine-Tuning proposé par certains fournisseurs. Voir [Données synthétiques](#donnees-synthetiques).

## RL

Reinforcement Learning, apprentissage par renforcement : le modèle génère, une récompense note ses réponses, et il est poussé vers les mieux notées. Voir [RL en ligne](#rl-en-ligne).

## RLAIF

RL from AI Feedback : RLHF où les préférences ou les notes viennent d'un modèle juge plutôt que d'humains. Voir [RL en ligne](#rl-en-ligne).

## RLHF

RL from Human Feedback : SFT, puis modèle de récompense appris sur des préférences humaines, puis RL (PPO) contre ce modèle. Voir [RL en ligne](#rl-en-ligne).

## RLOO

REINFORCE Leave-One-Out : RL où chaque réponse est comparée à la moyenne des autres réponses du même groupe. Voir [RL en ligne](#rl-en-ligne).

## RLVR

RL with Verifiable Rewards : RL dont la récompense vient d'un programme qui vérifie la réponse (résultat numérique, tests de code). Voir [RL en ligne](#rl-en-ligne).

## ROME

Rank-One Model Editing : modifie un fait en ajoutant une mise à jour de rang un à une couche MLP. Voir [Édition de connaissances](#edition-connaissances).

## RoPE

Rotary Position Embedding : encode la position des tokens par des rotations ; la plupart des extensions de contexte en modifient les fréquences. Voir [Extension de contexte](#extension-contexte).

## RRF

Reciprocal Rank Fusion : fusionne plusieurs classements en sommant l'inverse des rangs. Voir [RAG](#rag).

## Safetensors

Format de fichier des poids de modèle, rapide à charger et sans exécution de code, standard du Hugging Face Hub. Voir [PEFT, exemple du dépôt](#peft-exemple).

## Self-consistency

Échantillonner plusieurs raisonnements et garder la réponse la plus fréquente (vote majoritaire). Voir [Calcul à l'inférence](#calcul-inference).

## sentence-transformers

Bibliothèque d'encodeurs qui calculent des embeddings de phrases et des scores de pertinence (cross-encoders). Voir [RAG](#rag).

## SFT

Supervised Fine-Tuning : entraîner un modèle sur des paires instruction-réponse écrites à l'avance. La [LoRA](#peft) en est la version économe. Voir [Fine-tuning supervisé](#fine-tuning-supervise).

## SimPO

Simple Preference Optimization : variante de DPO sans modèle de référence, qui utilise la vraisemblance moyenne par token et une marge. Voir [Optimisation des préférences](#optimisation-preferences).

## SLERP

Interpolation sphérique entre les poids de deux modèles, qui préserve mieux leur norme qu'une moyenne. Voir [Fusion de modèles](#fusion-modeles).

## SOLAR, depth up-scaling

Agrandir un modèle en dupliquant une partie de ses couches, puis en le cicatrisant par pré-entraînement continu. Voir [Croissance d'architecture](#croissance-architecture).

## Sparsité 2:4

Élagage structuré qui garde 2 poids sur chaque groupe de 4, format accéléré matériellement par les GPU NVIDIA récents. Voir [Élagage](#elagage).

## STaR

Self-Taught Reasoner : le modèle génère des raisonnements, on garde ceux qui mènent à la bonne réponse, et on l'entraîne dessus, en plusieurs tours. Voir [Données synthétiques](#donnees-synthetiques).

## Température

Réglage de l'échantillonnage : à 0, le modèle prend toujours le token le plus probable (décodage glouton) ; plus haute, ses réponses varient davantage. Voir [Calcul à l'inférence](#calcul-inference).

## TIES

TrIm, Elect Sign & merge : fusion qui garde les plus fortes valeurs de chaque vecteur de tâche, élit un signe par coordonnée et ne moyenne que les valeurs d'accord. Voir [Fusion de modèles](#fusion-modeles).

## TOFU

Jeu de test du désapprentissage : des auteurs fictifs dont on fait apprendre puis oublier les biographies. Voir [Désapprentissage](#desapprentissage).

## Token

Morceau de texte (mot, partie de mot, ponctuation) que le modèle lit et produit ; les longueurs de contexte et les coûts se comptent en tokens. Voir [Extension de vocabulaire](#extension-vocabulaire).

## Tokenizer

Composant qui découpe le texte en tokens et les convertit en identifiants numériques. Voir [Extension de vocabulaire](#extension-vocabulaire).

## Trainer

Classe de transformers qui gère la boucle d'entraînement (lots, optimiseur, sauvegardes) ; TRL en dérive SFTTrainer, DPOTrainer et GRPOTrainer. Voir [Fine-tuning supervisé](#fine-tuning-supervise).

## transformers

Bibliothèque Hugging Face qui charge, exécute et entraîne la plupart des modèles du Hub. Voir [PEFT, exemple du dépôt](#peft-exemple).

## TRL

Transformer Reinforcement Learning : bibliothèque Hugging Face pour le SFT, les préférences (DPO...) et le RL (GRPO, PPO). Voir [RL en ligne](#rl-en-ligne).

## Vecteur de tâche

Différence entre les poids d'un modèle affiné et ceux de sa base : ce que le fine-tuning a ajouté. On peut additionner ou retrancher ces vecteurs. Voir [Fusion de modèles](#fusion-modeles).

## Vérificateur

Programme qui dit si une réponse est juste (calcul du résultat, exécution de tests) ; il sert de récompense en RL et de filtre en rejection sampling. Voir [RL en ligne](#rl-en-ligne).

## ViT

Vision Transformer : transformer qui lit une image découpée en carrés traités comme des tokens. Voir [Fine-tuning supervisé](#fine-tuning-supervise).

## vLLM

Serveur d'inférence rapide pour GPU, qui gère de nombreuses requêtes en parallèle et plusieurs adaptateurs LoRA. Voir [PEFT](#peft).

## VLM

Vision-Language Model : modèle qui lit des images et du texte et répond en texte (Qwen2.5-VL, LLaVA). Voir [Fine-tuning supervisé](#fine-tuning-supervise).

## W4A16, W8A8

Notation de la quantification : poids sur 4 bits et activations sur 16 bits, ou poids et activations sur 8 bits. Voir [Quantification](#quantification).

## Wanda

Pruning by Weights and activations : élagage qui retire les poids dont le produit valeur × activation moyenne est le plus faible, sans réentraînement. Voir [Élagage](#elagage).

## wikitext

Corpus d'articles Wikipédia en anglais, utilisé dans le dépôt pour mesurer la perplexité générale et comme textes de calibration. Voir [Quantification](#quantification).

## YaRN

Yet another RoPE extensioN : extension de contexte qui modifie les fréquences RoPE par bandes et ajuste la température de l'attention, avec un court fine-tuning. Voir [Extension de contexte](#extension-contexte).

## ZeRO

Optimisation de DeepSpeed qui répartit état de l'optimiseur, gradients et poids entre GPU (étapes 1 à 3), comme FSDP. Voir [Optimisation des préférences](#optimisation-preferences).

## Abliteration

Retirer d'un modèle la direction d'activation associée aux refus, en la projetant hors de ses poids. Voir [Pilotage par activations](#pilotage-activations).

## Block expansion, LLaMA Pro

Agrandir un modèle en intercalant des blocs initialisés pour ne rien changer au départ, puis en n'entraînant que ces nouveaux blocs. Voir [Croissance d'architecture](#croissance-architecture).

## Constitutional AI

Le modèle critique et révise ses propres réponses selon une liste de principes écrits ; les réponses révisées servent ensuite à l'entraîner. Voir [Données synthétiques](#donnees-synthetiques).

## CounterFact

Jeu de test de l'édition de connaissances : des faits contrefactuels à implanter, avec des questions pour vérifier la généralisation et la spécificité. Voir [Édition de connaissances](#edition-connaissances).

## Décodage glouton

Génération qui prend toujours le token le plus probable (température 0) : une seule réponse, reproductible. Voir [Calcul à l'inférence](#calcul-inference).

## Décodeur

Modèle qui génère le texte token par token en ne regardant que les tokens précédents ; c'est l'architecture des LLM. Voir [Fine-tuning supervisé](#fine-tuning-supervise).

## DeepSpeed

Bibliothèque de Microsoft pour entraîner de grands modèles sur plusieurs GPU, notamment avec ZeRO. Voir [Optimisation des préférences](#optimisation-preferences).

## Diffusion (modèle de)

Modèle qui génère une image en partant d'un bruit et en le débruitant pas à pas (Stable Diffusion, SDXL, Flux). Voir [PEFT](#peft).

## DreamBooth

Apprendre un sujet précis (une personne, un objet) à un modèle de diffusion à partir de quelques images, souvent avec une LoRA. Voir [PEFT](#peft).

## Encodeur-décodeur, seq2seq

Architecture où un encodeur lit l'entrée et un décodeur génère la sortie (T5, BART, Whisper) : traduction, résumé, transcription. Voir [Fine-tuning supervisé](#fine-tuning-supervise).

## FT-L

Fine-tuning contraint d'une seule couche pour corriger un fait ; référence de comparaison de l'édition de connaissances, moins spécifique que ROME. Voir [Édition de connaissances](#edition-connaissances).

## GSPO

Group Sequence Policy Optimization : variante de GRPO qui calcule les ratios d'importance sur toute la séquence plutôt que par token. Voir [RL en ligne](#rl-en-ligne).

## imatrix

Matrice d'importance calculée par llama.cpp sur un texte de calibration, pour mieux répartir les erreurs d'arrondi d'une quantification GGUF. Voir [Quantification](#quantification).

## IPO, CPO

Variantes de DPO : IPO remplace la perte sigmoïde par une perte quadratique, moins sujette au surapprentissage ; CPO se passe de modèle de référence et ajoute un terme de vraisemblance. Voir [Optimisation des préférences](#optimisation-preferences).

## JSD

Divergence de Jensen-Shannon : écart symétrique entre deux distributions, alternative à la divergence KL en distillation. Voir [Distillation](#distillation).

## KD

Knowledge Distillation : distillation de connaissances, en particulier sur les distributions de probabilités (logits) de l'enseignant. Voir [Distillation](#distillation).

## k-quants

Formats de quantification de llama.cpp (Q4_K_M, Q5_K_M...) qui regroupent les poids en blocs et super-blocs, chacun avec son échelle. Voir [Quantification](#quantification).

## llm-compressor

Bibliothèque du projet vLLM pour quantifier un modèle (GPTQ, AWQ, FP8) dans un format directement servi par vLLM. Voir [Quantification](#quantification).

## Log-probabilité

Logarithme de la probabilité qu'un modèle attribue à un texte ou à un token ; la somme sur une réponse sert à comparer des réponses ou à calculer la perplexité. Voir [Optimisation des préférences](#optimisation-preferences).

## LongRoPE

Extension de contexte qui cherche, par recherche évolutionnaire, un facteur d'étirement RoPE différent pour chaque fréquence. Voir [Extension de contexte](#extension-contexte).

## MEND, GRACE

Méthodes d'édition de connaissances : MEND apprend un petit réseau qui transforme le gradient en modification ; GRACE stocke les corrections dans un codebook consulté à l'inférence. Voir [Édition de connaissances](#edition-connaissances).

## MIPROv2

Optimiseur de DSPy qui cherche ensemble les consignes et les exemples d'un prompt, par optimisation bayésienne. Voir [Optimisation du prompt](#optimisation-prompt).

## Model soup, WiSE-FT

Moyenne des poids de plusieurs fine-tunes d'un même modèle (model soup), ou du modèle d'origine et de sa version affinée (WiSE-FT), pour gagner en robustesse. Voir [Fusion de modèles](#fusion-modeles).

## Oracle

Borne haute d'une méthode de sélection : la question compte comme réussie si au moins une des N réponses est juste (pass@N). Voir [Calcul à l'inférence](#calcul-inference).

## RMU

Representation Misdirection for Unlearning : désapprentissage qui brouille les activations internes sur les sujets à oublier. Voir [Désapprentissage](#desapprentissage).

## RULER

Jeu de test des longs contextes, plus exigeant que le simple passkey : recherche de plusieurs aiguilles, traçage de variables, agrégation. Voir [Extension de contexte](#extension-contexte).

## Self-Instruct

Faire écrire par un modèle de nouvelles consignes et leurs réponses à partir de quelques exemples, pour constituer un jeu de SFT. Voir [Données synthétiques](#donnees-synthetiques).

## Self-refine

Le modèle relit sa réponse, la critique, puis la corrige, en un ou plusieurs tours. Voir [Calcul à l'inférence](#calcul-inference).

## SparseGPT, ShortGPT

Méthodes d'élagage : SparseGPT met à zéro des poids en corrigeant les poids restants ; ShortGPT retire des couches entières jugées redondantes. Voir [Élagage](#elagage).

## SST-2

Jeu de critiques de films à classer en positive ou négative, référence de la classification de texte. Voir [Fine-tuning supervisé](#fine-tuning-supervise).

## Taux d'apprentissage

Taille du pas de mise à jour des poids à chaque itération. Trop haut, l'entraînement diverge ; trop bas, il stagne. Il monte souvent progressivement au début (warmup). Voir [Fine-tuning supervisé](#fine-tuning-supervise).

## TEI

Text Embeddings Inference : serveur de Hugging Face qui calcule embeddings et scores de reranking sur GPU, pour un RAG en production. Voir [RAG](#rag).

## torchao

Bibliothèque de PyTorch pour la quantification (int8, int4, FP8) et l'entraînement conscient de la quantification. Voir [Quantification](#quantification).

## Tülu 3

Recette ouverte de post-entraînement de l'Allen Institute : SFT, DPO puis RL à récompense vérifiable, avec données et code publiés. Voir [Quelle méthode pour quel besoin](#choisir-une-methode).

## Upcycling (MoE)

Transformer un modèle dense en mixture of experts en copiant ses couches MLP en plusieurs experts, puis en entraînant le routeur. Voir [Croissance d'architecture](#croissance-architecture).

## Vecteur de pilotage

Direction dans l'espace des activations, calculée à partir de paires de textes opposés, qu'on ajoute pendant la génération pour orienter un trait. Voir [Pilotage par activations](#pilotage-activations).

## Vote majoritaire

Garder la réponse donnée le plus souvent parmi plusieurs générations. Voir [Calcul à l'inférence](#calcul-inference).

## WER

Word Error Rate : part de mots mal transcrits, mesure de référence de la reconnaissance vocale. Voir [Fine-tuning supervisé](#fine-tuning-supervise).

## Whisper

Modèle encodeur-décodeur d'OpenAI pour la transcription de la parole. Voir [Fine-tuning supervisé](#fine-tuning-supervise).

## WMDP

Jeu de questions sur des connaissances dangereuses (biosécurité, cybersécurité, chimie) qui sert à mesurer le désapprentissage. Voir [Désapprentissage](#desapprentissage).
