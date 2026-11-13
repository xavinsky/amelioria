# Backlog

Référence des critères : [docs/cahier-des-charges.md](docs/cahier-des-charges.md). Avancement et notes d'environnement : [session-status.md](session-status.md). Point de reprise : [TODONEXT.md](TODONEXT.md).

## État

- Les 30 exemples (20 méthodes + variantes) s'exécutent en `--quick` et en configuration complète sur le poste local (RTX 5070 Ti Laptop, 11,5 Gio utilisables). Toutes les valeurs des sélecteurs passent en `--quick` (`scripts/selecteurs.sh`).
- Les configs complètes tiennent en mémoire (gradient checkpointing, micro-lots, AdamW 8 bits où nécessaire).
- Documentation transverse en place : `docs/transformers.md`, `docs/schemas-methodes.md` (36 schémas Mermaid validés), `docs/bibliotheques.md`, `docs/references.md` (181 papiers vérifiés), `docs/cahier-des-charges.md`.
- Rien n'est commité.

GPU réservé à une autre tâche de Xavier : aucun calcul GPU avant sa confirmation.

## Exécution sur ce poste

- Lancer les calculs longs détachés (`setsid nohup ... &`) et suivre les journaux : le garde-fou mémoire de Claude Code arrête les tâches d'arrière-plan quand le swap est chargé par d'autres applications, même avec ~22 Go de RAM disponibles.
- Décharger les modèles Ollama avant un entraînement (`keep_alive: 0`).
- Passe complète : `COMPLET=1 bash scripts/smoke.sh [NN]`, journaux dans `logs/complet/`. Tests rapides : `bash scripts/smoke.sh`. GPTQ/AWQ : `.venv-llmcompressor/bin/python 15-quantification/run.py --set methode=gptq`.

## Grille des exemples

Catégorie, statut (Démontré / Non prouvé / Non testé), résultat mesuré et réserve de chaque exemple : [exemples.yaml](exemples.yaml), source unique affichée par `make liste` et reprise dans le README (`make readme`). Bilan : 15 démontrés, 15 non prouvés ; versions `infra/` non testées. La tâche fil rouge arithmétique est **saturée** pour Qwen2.5-0.5B-Instruct (0,955 d'exactitude avec 256 tokens de génération) : les exemples qui l'utilisent pour enseigner une compétence ne peuvent rien démontrer.

## 1. Terminer la passe finale en cours

- [x] Relever les résultats de 12 à 15 (`logs/complet/`) dans `exemples.yaml`.
- [x] 15 GGUF et 16 à 20 en complet, FORM13 et EXEC17 relancés, test rapide de DATA02 audio-whisper : résultats dans `exemples.yaml`.

## 2. Outils de preuve (socle)

- [ ] `evaluation.exactitude` : intervalle de confiance à 95 % (Wilson) dans `resultats.json` et le tableau.
- [ ] Option de répétition sur plusieurs graines (moyenne ± écart) pour les entraînements courts.
- [ ] Témoin intégré là où l'effet peut avoir une autre cause (13, 14).

## 3. Refondre les exemples sur de vrais besoins

Relire les papiers de chaque méthode (`docs/references.md`) avant d'implémenter, vérifier que le modèle de départ échoue, puis appliquer la grille.

- [ ] **Tâche commune « routage de demandes clients »** : banking77 (77 intentions, 10 003 exemples d'entraînement, 3 080 de test), sortie = intention. Un LLM instruct en zéro-shot se trompe souvent ; c'est un besoin courant.
  - 02 SFT : SFT complet de Qwen2.5-0.5B-Instruct ; même jeu pour la variante encodeur (comparaison LLM / encodeur).
  - 03 PEFT : LoRA sur le même jeu, comparée au SFT complet à données égales (Biderman et al. : LoRA apprend moins mais oublie moins ; mesurer aussi la non-régression).
- [ ] **Raisonnement vérifiable** : GSM8K (7 473 / 1 319), tâche des papiers de référence.
  - 05 RL : GRPO/DAPO sur GSM8K avec récompense vérifiable (DeepSeekMath, DAPO) ; test complet 1 319.
  - 07 données synthétiques : rejection sampling fine-tuning sur GSM8K (RFT, STaR).
  - 06 distillation : enseignant nettement meilleur que l'élève sur GSM8K (mode `sequence` façon R1-Distill, puis `logits`) ; comprendre la régression mesurée sur l'arithmétique (décalage entraînement / génération, GKD).
- [ ] **04 préférences** : jeu de préférences réel (UltraFeedback binarisé ou HelpSteer), évaluation par un modèle de récompense externe (taux de victoire contre le modèle de départ) et non-régression (GSM8K ou IFEval).
- [ ] **01 connaissance interne** : suivre *Physics of Language Models 3.1* : entraînement mixte (documents de tous les personnages + questions-réponses d'une moitié), mesure de l'extraction sur l'autre moitié, avec plusieurs formulations par fait.
- [ ] **13 vocabulaire** : corpus réel mal tokenisé (langue ou domaine), mesures : tokens, bits par caractère, débit (Chinese-LLaMA, FOCUS). Le témoin à budget égal existe : sur les noms inventés, il atteint 0,41 bit par caractère contre 0,42, le gain est la compression seule.
- [ ] **14 croissance** : le témoin à 600 pas fait mieux que le modèle agrandi (14,3 contre 14,9). Essayer `expansion_blocs` (LLaMA Pro, l'original est préservé) et un budget de cicatrisation plus grand, puis comparer au témoin au même budget.
- [ ] **08 fusion multi-compétences** : fusionner deux fine-tunes réels de Qwen2.5-0.5B-Instruct produits par le dépôt (routage banking77 et GSM8K), mesurer les deux tâches (TIES, DARE).
- [ ] **02 seq2seq** : tâche naturellement seq2seq (traduction ou résumé de domaine, métrique chrF/ROUGE) où T5 de départ échoue.
- [ ] **02 Whisper** : audio où whisper-small se trompe nettement (vocabulaire spécialisé, accent, bruit téléphonique) ; jeu public à identifier.
- [ ] **09, 11** : élargir les jeux d'évaluation (plus de faits, plus de consignes) et afficher les IC.
- [ ] **16** : vérifier la récupération de la tâche avec plus de pas de distillation ou un élagage 2:4 (Minitron, Wanda).

## 4. Documentation après validation

- [ ] Section « Résultats mesurés » dans chaque README (tableau avec IC, matériel, durée, date).
- [ ] Mettre à jour « Ce que fait l'exemple », commandes et schémas des exemples refondus.
- [ ] Avertissement du `README.md` principal limité aux parties non testées : tous les `infra/` (multi-GPU, Slurm, vLLM, verl, mergekit, EasyEdit, open-unlearning, axolotl, Qdrant/TEI).

## 5. Pistes hors périmètre

- [ ] Tests unitaires sans modèle pour `commun/`, `08-fusion-modeles/fusion.py`, `16-elagage/elagage.py`, `14-croissance-architecture/croissance.py`.
- [ ] Méthode ou variante VLM (SFT/DPO multimodal).
- [ ] Données fil rouge en français (`donnees.langue`).
