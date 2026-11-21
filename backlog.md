# Backlog

Référence des critères : [docs/cahier-des-charges.md](docs/cahier-des-charges.md). Avancement et notes d'environnement : [session-status.md](session-status.md). Point de reprise : [TODONEXT.md](TODONEXT.md).

## État

- Les 30 exemples (20 méthodes + variantes) s'exécutent en `--quick` et en configuration complète sur le poste local (RTX 5070 Ti Laptop, 11,5 Gio utilisables). Toutes les valeurs des sélecteurs passent en `--quick` (`scripts/selecteurs.sh`).
- Les configs complètes tiennent en mémoire (gradient checkpointing, micro-lots, AdamW 8 bits où nécessaire).
- Documentation transverse en place : `docs/transformers.md`, `docs/schemas-methodes.md` (36 schémas Mermaid validés), `docs/bibliotheques.md`, `docs/references.md` (182 papiers vérifiés), `docs/cahier-des-charges.md`.
- Dépôt public github.com/xavinsky/amelioria : un push sur `main` republie le site sur tekio.org/amelioria/.

GPU partagé entre les projets de Xavier : tout calcul passe par le skill `gpu-local`.

## Exécution sur ce poste

- Lancer les calculs longs détachés (`setsid nohup ... &`) et suivre les journaux : le garde-fou mémoire de Claude Code arrête les tâches d'arrière-plan quand le swap est chargé par d'autres applications, même avec ~22 Go de RAM disponibles.
- Décharger les modèles Ollama avant un entraînement (`keep_alive: 0`).
- Passe complète : `COMPLET=1 bash scripts/smoke.sh [NN]`, journaux dans `logs/complet/`. Tests rapides : `bash scripts/smoke.sh`. GPTQ/AWQ : `.venv-llmcompressor/bin/python 15-quantification/run.py --set methode=gptq`.

## Grille des exemples

Catégorie, statut (Démontré / Non prouvé / Non testé), résultat mesuré et réserve de chaque exemple : [exemples.yaml](exemples.yaml), source unique affichée par `make liste` et reprise dans le README (`make readme`). Bilan : 25 démontrés, 5 non prouvés ; versions `infra/` non testées. La tâche fil rouge arithmétique est **saturée** pour Qwen2.5-0.5B-Instruct (0,955 d'exactitude avec 256 tokens de génération) : les exemples qui l'utilisent pour enseigner une compétence ne peuvent rien démontrer.

## 1. Terminer la passe finale en cours

- [x] Relever les résultats de 12 à 15 (`logs/complet/`) dans `exemples.yaml`.
- [x] 15 GGUF et 16 à 20 en complet, FORM13 et EXEC17 relancés, test rapide de DATA02 audio-whisper : résultats dans `exemples.yaml`.

## 2. Outils de preuve (socle)

- [x] Intervalle de confiance à 95 % (Wilson) et comparaisons appariées (bootstrap, McNemar) dans `resultats.json` et le tableau (`commun/preuve.py`).
- [x] Répétition sur plusieurs graines : `make graines` (moyenne ± écart type).
- [x] Témoin intégré là où l'effet peut avoir une autre cause (13, 14).

## 3. Refondre les exemples sur de vrais besoins

Relire les papiers de chaque méthode (`docs/references.md`) avant d'implémenter, vérifier que le modèle de départ échoue, puis appliquer la grille.

- [x] **Tâche commune « routage de demandes clients »** : banking77 (77 intentions, 10 003 exemples d'entraînement, 3 080 de test), liste des intentions en message système, non-régression sur l'arithmétique, remélange de 15 % de demandes générales (réponses du modèle d'origine).
  - 02 SFT complet : 0,231 → 0,904 ; non-régression 0,955 → 0,765 (0,025 sans remélange). Variante encodeur ModernBERT : 0,937 en 85 s.
  - 03 LoRA : 0,229 → 0,891 ; non-régression 0,955 → 0,95 ; adaptateur désactivé 0,955.
- [x] **17 prompt sur banking77** : zéro-shot 0,229 ; consigne OPRO 0,263 ; 8 exemples les plus proches 0,816 ; témoin sans LLM (vote des voisins) 0,920.
- [ ] **Raisonnement vérifiable** : GSM8K (7 473 / 1 319), tâche des papiers de référence.
  - [x] 05 RL : DAPO sur 800 problèmes, récompense vérifiable : 0,411 → 0,465 (p = 0,0002), non-régression inchangée.
  - [ ] 07 données synthétiques : depuis l'instruct, RFT + STaR 0,411 → 0,393 (n.s.). Depuis Qwen2.5-0.5B de base amorcé, plafond ReST-EM : 0,363 → 0,402 (p = 0,004), témoin 0,332, une graine à remesurer (génération qui continuait après « Answer: ») sur 3 graines.
  - [x] 06 distillation : Qwen2.5-3B → 0.5B en mode `sequence` sur GSM8K : 0,411 → 0,451 ; témoin sur solutions humaines 0,302 ; enseignant 0,760.
- [ ] **04 préférences** : DPO LoRA sur 16 000 paires UltraFeedback, juge Skywork-Reward-V2-1.7B : victoires contre le départ 0,552 [0,508 ; 0,595], témoin SFT 0,380, non-régression stable. Sur 3 graines : 0,533 ± 0,032, net sur 2 graines sur 3. Piste : modèle non aligné, ou DPO itératif. Variante DPO itératif (paires produites par le modèle, 3 tours) : 0,616 [0,573 ; 0,658] sur une graine, 2 graines à lancer.
- [x] **13 vocabulaire** : Wikipédia en breton, 1 000 tokens, 3 graines : −15,8 % de tokens ; 1,602 bit par caractère contre 1,572 pour le témoin (qualité 2 % en dessous à entraînement égal).
- [x] **14 croissance** : LLaMA Pro (8 couches ajoutées, seules entraînées) sur la Wikipédia en breton, 3 graines : anglais 19,9 → 22,0 ; témoin entraîné en entier, au même niveau en breton : 23,6 (écart 1,57 ± 0,21).
- [ ] **08 fusion multi-compétences** : fusionner deux fine-tunes réels de Qwen2.5-0.5B-Instruct produits par le dépôt (routage banking77 et GSM8K), mesurer les deux tâches (TIES, DARE).
- [x] **02 seq2seq** : résumé de dialogues DialogSum, Flan-T5-base : ROUGE-L 0,291 → 0,432 (+0,141 [+0,127 ; +0,156]).
- [x] **02 Whisper** : EdAcc (40 accents), whisper-small : WER 0,232 → 0,181 sur 3 graines ; LibriSpeech 0,034 → 0,041.
- [x] **01 connaissance interne** : protocole *Physics of LM 3.1*, 400 personnages, extraction sur les 200 jamais questionnés 0,001 → 0,9996 sur 3 graines ; témoin à une biographie 0,997.
- [ ] **01 variante encodeur** : la perte de masquage (2,38 → 1,09) ne mesure aucune tâche finale. Protocole banking77 : masquage continu sur les messages sans étiquettes, classification avec 10 exemples par intention, témoin sans masquage, 3 graines.
- [ ] **09, 11** : élargir les jeux d'évaluation (plus de faits, plus de consignes) et afficher les IC.
- [x] **03 diffusion** : mesures DINO et CLIP du papier DreamBooth, 4 prompts × 8 graines appariés : sujet 0,281 → 0,751, prompt 0,300 → 0,293 (n.s.).
- [ ] **16 élagage** : Wanda 50 % + distillation de 1 000 pas sur Wikipédia et conversations du modèle d'origine : tâche 0,88 → 0,34 → 0,66 ; 2:4 : 0,38. Récupération partielle, pas de gain de vitesse. Structuré : largeur 0,14, couches 0,09 (vitesse +8 %). Conclusion documentée : ne paie pas à 0,5B ; plans possibles sur le poste dans `estimations.md` (jusqu'à 3B avec cicatrisation, 7B sans).

## 4. Documentation après validation

- [ ] Section « Résultats mesurés » dans chaque README (tableau avec IC, matériel, durée, date).
- [ ] Mettre à jour « Ce que fait l'exemple », commandes et schémas des exemples refondus.
- [ ] Avertissement du `README.md` principal limité aux parties non testées : tous les `infra/` (multi-GPU, Slurm, vLLM, verl, mergekit, EasyEdit, open-unlearning, axolotl, Qdrant/TEI).

## 5. Pistes hors périmètre

- [ ] Tests unitaires sans modèle pour `commun/`, `08-fusion-modeles/fusion.py`, `16-elagage/elagage.py`, `14-croissance-architecture/croissance.py`.
- [ ] Méthode ou variante VLM (SFT/DPO multimodal).
- [ ] Données fil rouge en français (`donnees.langue`).
