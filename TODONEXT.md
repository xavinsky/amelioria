# TODONEXT — reprise du projet

Point d'entrée pour reprendre le travail. Lire ce fichier, puis les documents listés dans « À lire avant d'agir ».

## Objectif du projet

Un panorama exécutable des méthodes pour améliorer un modèle d'IA : 20 dossiers, chacun avec un exemple local, des variantes et une version grosse infra. Chaque exemple doit traiter **un besoin réellement utile en entreprise**, avec **le bon concept**, une **implémentation limpide**, et un **résultat prouvé sans ambiguïté** : intervalle de confiance, plusieurs graines, témoin, non-régression. Pas de démonstration fabriquée ni de réglage ad hoc pour obtenir le chiffre voulu. Si le résultat n'apparaît pas, l'approche ou l'implémentation est mauvaise : relire les papiers de la méthode et se réorienter. Code, configs et docs décrivent l'état actuel, jamais leur historique.

## À lire avant d'agir

1. [docs/cahier-des-charges.md](docs/cahier-des-charges.md) : critères et grille de validation d'un exemple.
2. [exemples.yaml](exemples.yaml) (`make liste`) : catégories, statut, résultat et prérequis de chaque exemple ; [backlog.md](backlog.md) : plan détaillé.
3. [session-status.md](session-status.md) : avancement (journal daté des étapes), environnement, faits techniques établis, journaux.
4. [README.md](README.md), [docs/structure-commune.md](docs/structure-commune.md), puis le README et le `run.py` du dossier sur lequel on travaille.
5. [docs/references.md](docs/references.md) : papiers à relire avant de refondre une méthode.

## Conditions de travail

- **GPU (RTX 5070 Ti Laptop, 11,5 Gio utilisables)** : partagé entre les projets de Xavier, tout calcul passe par le skill `gpu-local`.
- Calculs longs : lancer détachés (`setsid nohup bash -c '...' > journal 2>&1 &`) puis sonder le journal. Les tâches d'arrière-plan de Claude Code sont arrêtées par le garde-fou mémoire à cause du swap des autres applications, alors que la RAM reste disponible.
- Décharger les modèles Ollama avant un entraînement.
- Installation d'un poste : `TORCH_INDEX_URL=https://download.pytorch.org/whl/cu130 bash scripts/installer.sh --extras`, puis `bash scripts/installer.sh --llmcompressor` (déjà fait sur ce poste).
- Commandes : `make test EX=NN [VAR=...]`, `make run EX=NN`, `make smoke`, `make complet EX=NN` (journaux dans `logs/complet/`), `make selecteurs EX=NN`, `make help`. Sur ce poste, préfixer les calculs GPU par `LANCEUR="gpu-local executer --motif ameliorer-modele-ia --"`.
- Tout changement de statut ou de résultat se fait dans `exemples.yaml`, puis `make readme`.
- Donner à Xavier un point d'avancement après chaque exemple validé ou en échec.
- **Tracer l'avancement dans [session-status.md](session-status.md)**, section « Avancement » : une ligne datée par étape terminée, exemple validé ou en échec (résultat chiffré, journal associé). Tenir aussi à jour la grille de `backlog.md` et les cases de ce fichier.
- Ne commiter et ne pousser qu'à la demande de Xavier (messages courts, sans ligne de co-auteur Claude). Le dépôt public est github.com/xavinsky/amelioria ; un push sur `main` republie le site sur tekio.org/amelioria/.

## État

- **Exécution** : les 30 exemples tournent en `--quick` et en complet ; toutes les valeurs des sélecteurs passent en `--quick` ; les configs complètes tiennent en mémoire.
- **Documentation transverse** : `docs/transformers.md`, `docs/schemas-methodes.md` (36 schémas Mermaid validés), `docs/bibliotheques.md`, `docs/references.md` (182 papiers vérifiés sur arXiv), `docs/cahier-des-charges.md`.
- **Passe complète finale** (code et configs actuels, `logs/complet_final.txt`, un journal par exemple dans `logs/complet/`) : 01 à 15 terminés sans erreur ; arrêtée pour libérer le GPU avant 15 variante GGUF, 16, 17, 18, 19 et 20.
- **Catalogue** (`exemples.yaml`, `make liste`) : 5 catégories (données, signal, chirurgie, remodeler, sans toucher aux poids) ; 24 exemples démontrés, 6 non prouvés, versions `infra/` non testées.
- **Commandes** : `Makefile` racine et un par dossier ; chaque `make test/run` vérifie d'abord les prérequis. Non encore exécuté de bout en bout sur un venv neuf.
- **Cause principale des refontes** : la tâche fil rouge arithmétique est déjà résolue par Qwen2.5-0.5B-Instruct (0,955) et 1,5B-Instruct (0,995). Les exemples qui l'utilisent pour enseigner une compétence ne démontrent rien. Pour 13 et 14, l'effet n'est pas isolé faute de témoin.

## Prochaines étapes

### 1. Clore la passe finale

- [x] Relever les résultats déjà produits de 12, 13, 14 et 15 et les reporter dans `exemples.yaml`.
- [x] FORM13 et EXEC17 relancés, 15 GGUF et 16 à 20 en complet, test rapide de 02 audio-whisper : résultats et durées dans `exemples.yaml` (journaux : `logs/complet/`, `logs/relance.txt`, `logs/test/`).

### 2. Site AmeliorIA et installation de bout en bout

- [x] Site AmeliorIA (`site/`, `make site`), publié sur tekio.org/amelioria/ par GitHub Pages : arbre de décision, une page Concept et une page Exemple de code par méthode, catégories, statut et travail en cours, lexique, liens vers le code. Conventions : `site/README.md`.
- [ ] Relire avec Xavier le contenu des six graphes de décision (`site/content/choisir/choisir/0[2-7]-cas-*.md`) et le classement `acces` / `donnees` / `calcul` / `duree` de `exemples.yaml`.
- [ ] Vérifier `make install EXTRAS=1` sur un venv neuf, puis `make smoke` (GPU).

### 3. Outils de preuve dans le socle

- [x] Intervalle de confiance à 95 % (Wilson) sur les exactitudes et comparaisons appariées (bootstrap, McNemar), dans `resultats.json` et le tableau du rapport (`commun/preuve.py`).
- [x] Répétition sur plusieurs graines : `make graines EX=… N=3`.
- [ ] Témoin intégré là où l'effet peut avoir une autre cause.

### 4. Refondre les exemples, dans cet ordre

Pour chacun : relire les papiers, vérifier que le modèle de départ échoue sur le besoin, implémenter, mesurer avec les outils de preuve, valider avec la grille, mettre à jour README et schéma.

Faits (résultats dans `exemples.yaml` et les README) :

- [x] **DATA02 SFT, DATA03 LoRA** sur banking77 : 0,23 → 0,90 et 0,89, non-régression et remélange ; variante encodeur 0,937.
- [x] **EXEC17 prompt**, même routage : few-shot par similarité 0,229 → 0,816 ; témoin sans LLM 0,920.
- [x] **FORM13 vocabulaire**, Wikipédia en breton, 3 graines : −15,8 % de tokens, qualité 2 % sous le témoin.
- [x] **SIGN06 distillation** sur GSM8K : 0,411 → 0,451 ; témoin sur solutions humaines 0,302.
- [x] **SIGN05 RL** DAPO sur GSM8K : 0,411 → 0,465, non-régression inchangée.
- [x] **DATA03 diffusion** : fidélité au sujet DINO 0,281 → 0,751, fidélité au prompt CLIP inchangée.
- [x] **DATA02 seq2seq** : résumé de dialogues, ROUGE-L 0,291 → 0,432.
- [x] **FORM14 croissance** : LLaMA Pro sur le breton, 3 graines : à niveau de breton égal, anglais 22,0 contre 23,6 pour le témoin.

À faire, dans cet ordre :

1. [ ] **SIGN07 données synthétiques** : sans gain sur Qwen2.5-0.5B-Instruct, en rejection sampling (0,387) comme en RFT + STaR (0,393). Reprendre sur Qwen2.5-0.5B de base amorcé en few-shot, avec un plafond de solutions par question (ReST-EM).
2. [ ] **DATA01 connaissance interne** : entraînement mixte de *Physics of Language Models 3.1* (documents de tous les personnages + questions-réponses d'une moitié), extraction mesurée sur l'autre moitié.
3. [ ] **SIGN04 préférences** : jeu de préférences réel (UltraFeedback ou HelpSteer), taux de victoire mesuré par un modèle de récompense externe, non-régression.
4. [ ] **Variante DATA02 Whisper** : audio où Whisper se trompe (accent, bruit).
5. [ ] **FORM16 élagage** : Wanda 50 % récupère la tâche à 0,66 sur 0,88 ; élagage structuré ou conclusion documentée.
6. [ ] **EDIT08 fusion multi-compétences** : fusionner le routage banking77 (DATA02) et le RL GSM8K (SIGN05), mesurer les deux tâches.
7. [ ] **EDIT09, EDIT11** : jeux d'évaluation plus larges et intervalles de confiance.

Mémoire : avec Qwen2.5-0.5B en fp32 sur 12 Go, les réponses de 512 tokens et le vocabulaire de 152 000 tokens imposent de petits lots (2 à 4), et AdamW 8 bits pour le RL. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` plante sur ce poste (NVML indisponible). Sans vLLM, une évaluation complète de GSM8K prend 5 à 15 minutes par modèle.

### 5. Documentation finale

- [ ] Section « Résultats mesurés » dans chaque README (avec IC, matériel, durée, date).
- [ ] Mettre à jour « Ce que fait l'exemple », commandes et schémas des exemples refondus.
- [ ] Limiter l'avertissement du `README.md` principal aux parties non testées (`infra/`).
- [ ] Vérifier que `exemples.yaml`, `backlog.md`, la section « Avancement » de `session-status.md` et ce fichier sont à jour.
