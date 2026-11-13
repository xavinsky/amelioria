# Notes de session

Mesures du 2026-09-15 sur RTX 5070 Ti Laptop (11,5 Gio utilisables), 30 Go de RAM. Plan de travail et grille des exemples : [backlog.md](backlog.md). Critères : [docs/cahier-des-charges.md](docs/cahier-des-charges.md).

## Avancement

Une ligne par étape terminée, exemple validé ou en échec, la plus récente en bas. Étapes à venir : [TODONEXT.md](TODONEXT.md).

| Date | Étape | Résultat | Journal |
|---|---|---|---|
| 2026-09-15 | Environnement : extras, `.venv-llmcompressor`, llama.cpp CPU | Prêt | — |
| 2026-09-15 | Tests rapides des 30 exemples | 30/30 | `logs/smoke_final.txt` (02 audio-whisper à relancer) |
| 2026-09-15 | Balayage des sélecteurs en `--quick` | 70/70 | `logs/selecteurs/` |
| 2026-09-15 | Mode API (17 à 20, Ollama) | 4/4 | `logs/api/` |
| 2026-09-15 | Calibrage mémoire des configs complètes | Toutes tiennent sur 11,5 Gio | `logs/calibrage/` |
| 2026-09-15 | Documentation transverse (transformers, schémas, bibliothèques, références, cahier des charges) | 36 schémas validés, 181 références vérifiées | `docs/` |
| 2026-09-15 | Grille des exemples après passe complète | 15 validés, 12 à refondre, 3 à vérifier | `backlog.md` |
| 2026-09-15 | Cahier des charges, `TODONEXT.md`, installeur aligné sur l'environnement testé, README à jour | Fait | `docs/cahier-des-charges.md`, `scripts/installer.sh` |
| 2026-09-15 | Passe complète finale | 01 à 15 terminés sans erreur ; arrêtée pour libérer le GPU (reste 15 GGUF, 16 à 20) | `logs/complet_final.txt` |
| 2026-09-15 | GPU rendu à Xavier pour une autre tâche | Aucun calcul GPU avant sa confirmation | — |
| 2026-11-12 | Résultats de 12 à 15 relevés | 12 démontré (perplexité 8k 422 → 19) ; 13, 14 non prouvés ; 15 démontré | `logs/complet/` |
| 2026-11-12 | Catalogue `exemples.yaml` : 5 catégories, statuts Démontré / Non prouvé / Non testé, prérequis | 15 démontrés, 15 non prouvés, `infra/` non testé | `make liste` |
| 2026-11-12 | `Makefile` racine + un par dossier, vérification des prérequis, section du README générée | `make help/liste/verifier/readme` testés ; `make test/run` non exécutés (GPU en pause) | — |
| 2026-11-12 | Site AmeliorIA (`site/`, moteur memoria-gcp) : arbre de décision, 3 fiches, installation, statut des exemples, lexique | Build sans erreur ; rendu vérifié en clair et en sombre (Chrome headless) | `make site` |
| 2026-11-12 | Arbre de décision : critère de choix en tête de case, enchaînements « si insuffisant », icônes de calcul et de temps (`calcul`, `duree` dans `exemples.yaml`), tableau des critères généré | Rendu vérifié | `make site` |
| 2026-11-12 | Six graphes de décision par cas (pages « Cas : … »), pictogrammes d'accès au modèle et de données à fournir, tableau des critères retiré | Build sans erreur, rendu vérifié en clair et en sombre | `make site` |
| 2026-11-12 | Pages de méthode générées depuis les README (Concept, Exemple), durées mesurées, schéma des composants et texte des étapes pour les 20 exemples | 20 schémas Mermaid rendus sans erreur, numérotation schéma/texte cohérente | `make site` |
| 2026-11-12 | 13 vocabulaire : initialisation par moyenne des nouveaux tokens corrigée (ids dans les lignes de réserve de la matrice) | Code corrigé, vérifié sur le tokenizer ; résultats à remesurer (GPU) | — |
| 2026-11-13 | Passe complète finale terminée : FORM13, EXEC17 à EXEC20, FORM16, FORM15 GGUF ; test rapide de DATA02 audio-whisper | 8/8 sans erreur ; résultats inchangés sauf FORM13 (bits par caractère 0,57 → 0,42 avec l'initialisation corrigée) | `logs/relance.txt`, `logs/complet/` |

## Environnement

| Élément | État |
|---|---|
| `.venv` | Socle + bitsandbytes 0.50, torchao 0.18, diffusers 0.40, torchcodec 0.16, torchvision 0.29 (cu130), sentencepiece, gguf 0.19, cmake |
| `.venv-llmcompressor` | llmcompressor 0.13, compressed-tensors 0.18, torch 2.13, transformers 5.14 (ignoré par git) |
| `~/llama.cpp` | Compilé CPU seul (pas de nvcc sur le poste) |
| Ollama | `localhost:11434/v1` : llama3.2:3b, llama3.1:8b (outils), qwen3:8b, qwen2.5:14b... |
| `nvidia-smi` | En échec (driver 610 / bibliothèque 615) ; torch.cuda fonctionne ; `expandable_segments` inutilisable (dépend de NVML) |
| Cache HF | ~50 Go, 196 Go libres sur `/home` |

## Faits techniques établis

Utiles pour toute nouvelle implémentation dans le dépôt.

- **Précision des mesures** : après `Trainer.train()`, accelerate laisse `model.forward` sous autocast. `commun.modeles.sans_autocast` le retire ; toute mesure écrite hors du socle doit l'appeler.
- **TRL 1.13** : toutes les configs activent `gradient_checkpointing` par défaut ; `commun.entrainement.arguments` le fixe à la valeur de la config. Prefix tuning est incompatible avec le checkpointing.
- **Mémoire** : SFT complet fp32 de 0,5B ≈ 8 Go fixes + logits (vocabulaire 152k). Libérer l'optimiseur (`entrainement.liberer`) avant un second entraînement ou une génération lourde. Pics calibrés : `logs/calibrage/`.
- **Évaluation** : 256 tokens de génération minimum sur les tâches à raisonnement (les réponses de Qwen instruct sont longues) ; perplexité générale sur le split `test` de wikitext.
- **YAML** : `--set` convertit `5e-6` en nombre et garde `2:4` en texte.
- **Quantification** : torchao int4 au format `tile_packed_to_4d` ; transformers décompresse GPTQ/AWQ à la première génération (mesurer la mémoire avant) ; dimensions multiples de 128 pour GPTQ/AWQ/int4.
- **Fusion** : vérifier l'écart relatif à la base (`fusion.ecart_relatif`) ; Qwen2.5-Math-1.5B-Instruct (100 %) et Coder-1.5B-Instruct (61 %) ne sont pas des fine-tunes fusionnables de Qwen2.5-1.5B (Instruct : 1,2 %).
- **DPO complet** sur des paires hors politique au format différent du modèle : la génération s'effondre (0,95 → 0,37) quel que soit l'optimiseur ; LoRA la préserve.
- **Modèle de récompense** : poids maîtres fp32 nécessaires (bf16 pur : 0,66 ; fp32 : 0,98).
- **T5** : découpage irrégulier des nombres, peu adapté à l'arithmétique.
- **Tâche fil rouge arithmétique** : Qwen2.5-0.5B-Instruct 0,955 et 1,5B-Instruct 0,995 ; ne convient plus pour démontrer l'apprentissage d'une compétence.
- **Harnais Claude Code** : les tâches d'arrière-plan sont tuées quand le swap est chargé ; lancer détaché (`setsid nohup`).

## Outils de travail (scratchpad de session)

`/tmp/claude-1000/-data-src-xavinsky-ameliorer-modele-ia/28848c8e-982c-4929-a9e5-8e49a368b1f1/scratchpad/` (dans `/tmp` : peut disparaître au redémarrage ; rien d'indispensable, les résultats sont dans `logs/` et `sorties/`) :

- `outils/pic.py` : exécute un `run.py` et affiche le pic de mémoire GPU ; `outils/essai.sh`, `outils/calibrer.sh`.
- `outils/resultat.sh fichier.log` : tableau de résultats et durées d'un journal.
- `outils/final.sh` : `smoke.sh` puis passe complète.
- `mermaid/valider.mjs` (parseur Mermaid sur les docs), `mermaid/rendre.sh` (rendu PNG et planches).
- `refs/arxiv.json` : métadonnées arXiv vérifiées des 172 papiers.
- `mesures.md` : relevé brut des mesures.

## Journaux

- `logs/smoke_final.txt`, `logs/complet_final.txt` : passe finale ; `logs/complet/*.log` : un journal par exemple.
- `logs/selecteurs/` : balayage des sélecteurs ; `logs/essais/` : essais ciblés (DPO, fusion, vocabulaire, distillation encodeur, OPRO) ; `logs/api/` : mode API.

## Fichiers ajoutés ou modifiés (non commités)

- Nouveaux : `site/`, `Makefile`, `NN-*/Makefile`, `commun/exemple.mk`, `exemples.yaml`, `scripts/catalogue.py`, `06-distillation/README.md`, `docs/{transformers,schemas-methodes,bibliotheques,references,cahier-des-charges}.md`, `scripts/selecteurs.sh`, `session-status.md`, `TODONEXT.md`.
- Socle : `commun/{config,donnees,entrainement,evaluation,generateur,modeles}.py`.
- Exemples : `run.py`, `config.yaml` et README de la plupart des dossiers ; `requirements.txt` (torchvision), `requirements-extras.txt`, `.gitignore` (`.venv-llmcompressor/`), `scripts/smoke.sh` (mode `COMPLET=1`), `scripts/installer.sh` (`TORCH_INDEX_URL`, `--llmcompressor`), `README.md`, `backlog.md`.
