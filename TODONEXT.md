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

- **GPU (RTX 5070 Ti Laptop, 11,5 Gio utilisables) : réservé à une autre tâche de Xavier. Ne lancer aucun calcul GPU (ni test, ni exemple) avant sa confirmation qu'il est de nouveau disponible.** La lecture, la conception et l'écriture de code restent possibles.
- Calculs longs : lancer détachés (`setsid nohup bash -c '...' > journal 2>&1 &`) puis sonder le journal. Les tâches d'arrière-plan de Claude Code sont arrêtées par le garde-fou mémoire à cause du swap des autres applications, alors que la RAM reste disponible.
- Décharger les modèles Ollama avant un entraînement.
- Installation d'un poste : `TORCH_INDEX_URL=https://download.pytorch.org/whl/cu130 bash scripts/installer.sh --extras`, puis `bash scripts/installer.sh --llmcompressor` (déjà fait sur ce poste).
- Commandes : `make test EX=NN [VAR=...]`, `make run EX=NN`, `make smoke`, `make complet EX=NN` (journaux dans `logs/complet/`), `make selecteurs EX=NN`, `make help`. Sur ce poste, préfixer les calculs GPU par `LANCEUR="gpu-local executer --motif ameliorer-modele-ia --"`.
- Tout changement de statut ou de résultat se fait dans `exemples.yaml`, puis `make readme`.
- Donner à Xavier un point d'avancement après chaque exemple validé ou en échec.
- **Tracer l'avancement dans [session-status.md](session-status.md)**, section « Avancement » : une ligne datée par étape terminée, exemple validé ou en échec (résultat chiffré, journal associé). Tenir aussi à jour la grille de `backlog.md` et les cases de ce fichier.
- Rien n'est commité : ne commiter qu'à la demande de Xavier (messages courts, sans ligne de co-auteur Claude).

## État

- **Exécution** : les 30 exemples tournent en `--quick` et en complet ; toutes les valeurs des sélecteurs passent en `--quick` ; les configs complètes tiennent en mémoire.
- **Documentation transverse** : `docs/transformers.md`, `docs/schemas-methodes.md` (36 schémas Mermaid validés), `docs/bibliotheques.md`, `docs/references.md` (181 papiers vérifiés sur arXiv), `docs/cahier-des-charges.md`.
- **Passe complète finale** (code et configs actuels, `logs/complet_final.txt`, un journal par exemple dans `logs/complet/`) : 01 à 15 terminés sans erreur ; arrêtée pour libérer le GPU avant 15 variante GGUF, 16, 17, 18, 19 et 20.
- **Catalogue** (`exemples.yaml`, `make liste`) : 5 catégories (données, signal, chirurgie, remodeler, sans toucher aux poids) ; 15 exemples démontrés, 15 non prouvés, versions `infra/` non testées.
- **Commandes** : `Makefile` racine et un par dossier ; chaque `make test/run` vérifie d'abord les prérequis. Non encore exécuté de bout en bout sur un venv neuf.
- **Cause principale des refontes** : la tâche fil rouge arithmétique est déjà résolue par Qwen2.5-0.5B-Instruct (0,955) et 1,5B-Instruct (0,995). Les exemples qui l'utilisent pour enseigner une compétence ne démontrent rien. Pour 13 et 14, l'effet n'est pas isolé faute de témoin.

## Prochaines étapes

### 1. Clore la passe finale

- [x] Relever les résultats déjà produits de 12, 13, 14 et 15 et les reporter dans `exemples.yaml`.
- [ ] Quand le GPU est de nouveau disponible : `setsid nohup bash -c 'COMPLET=1 bash scripts/smoke.sh 15 16 17 18 19 20 > logs/complet_final_suite.txt 2>&1' &` (le filtre `15` relance aussi 15, qui prend une minute), puis relever 15 GGUF et 16 à 20.
- [ ] Relancer `bash scripts/smoke.sh 02` : la section quick de `02 audio-whisper` a été corrigée après la passe `--quick`.

### 2. Site AmeliorIA et installation de bout en bout

- [x] Site AmeliorIA (`site/`, moteur de memoria-gcp, `make site`) : arbre de décision, fiches 03 PEFT, 15 quantification, 18 RAG, installation, statut des exemples, lexique. Conventions : `site/README.md`.
- [ ] Relire avec Xavier le contenu des six graphes de décision (`site/content/choisir/choisir/0[2-7]-cas-*.md`) et le classement `acces` / `donnees` / `calcul` / `duree` de `exemples.yaml`.
- [ ] Noms des auteurs des icônes Flaticon (attribution exigée par la licence gratuite) dans `site/content/refs/ressources/`.
- [ ] Fiches des 17 autres méthodes (schéma d'intégration, principe, quand l'utiliser, pièges, références), puis liens de l'arbre et des tableaux de « Quelle méthode pour quel besoin » vers ces fiches.
- [ ] Publication : emplacement (tekio.org/amelioria/ prévu dans `build.py`) et dépôt (dans ce dépôt ou en sous-module, comme memoria-gcp) à décider avec Xavier.
- [ ] Vérifier `make install EXTRAS=1` sur un venv neuf, puis `make smoke` (GPU).

### 3. Outils de preuve dans le socle

- [ ] Intervalle de confiance à 95 % (Wilson) sur les exactitudes, dans `resultats.json` et le tableau du rapport.
- [ ] Répétition sur plusieurs graines (moyenne ± écart) pour les entraînements courts.
- [ ] Témoin intégré là où l'effet peut avoir une autre cause.

### 4. Refondre les exemples, dans cet ordre

Pour chacun : relire les papiers, vérifier que le modèle de départ échoue sur le besoin, implémenter, mesurer avec les outils de preuve, valider avec la grille, mettre à jour README et schéma.

1. [ ] **02 SFT puis 03 LoRA — routage de demandes clients** (banking77 : 77 intentions, 3 080 exemples de test). SFT complet de Qwen2.5-0.5B-Instruct ; variante encodeur sur le même jeu ; LoRA comparée au SFT complet à données égales, avec non-régression.
2. [ ] **05 RL — raisonnement vérifiable** : GRPO/DAPO sur GSM8K avec récompense vérifiable, test complet (1 319 questions).
3. [ ] **07 données synthétiques** : rejection sampling fine-tuning sur GSM8K (RFT, STaR).
4. [ ] **06 distillation LLM** : GSM8K avec un enseignant nettement meilleur que l'élève ; comprendre la régression observée sur l'arithmétique (décalage entraînement / génération, GKD).
5. [ ] **04 préférences** : jeu de préférences réel (UltraFeedback ou HelpSteer), taux de victoire mesuré par un modèle de récompense externe, non-régression.
6. [ ] **01 connaissance interne** : entraînement mixte de *Physics of Language Models 3.1* (documents de tous les personnages + questions-réponses d'une moitié), extraction mesurée sur l'autre moitié.
7. [ ] **13 vocabulaire et 14 croissance** : témoins entraînés à budget égal (sans nouveaux tokens ; modèle d'origine sans agrandissement).
8. [ ] **08 fusion multi-compétences** : fusion de deux fine-tunes réels de Qwen2.5-0.5B-Instruct produits par 02 et 05, mesure des deux tâches.
9. [ ] **02 seq2seq et 02 Whisper** : tâches où le modèle de départ échoue réellement (traduction ou résumé de domaine ; audio à vocabulaire spécialisé, accent ou bruit).
10. [ ] **09, 11, 16** : jeux d'évaluation plus larges et IC ; vérifier la récupération de la tâche en 16.

### 5. Documentation finale

- [ ] Section « Résultats mesurés » dans chaque README (avec IC, matériel, durée, date).
- [ ] Mettre à jour « Ce que fait l'exemple », commandes et schémas des exemples refondus.
- [ ] Limiter l'avertissement du `README.md` principal aux parties non testées (`infra/`).
- [ ] Vérifier que `exemples.yaml`, `backlog.md`, la section « Avancement » de `session-status.md` et ce fichier sont à jour.
