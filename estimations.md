# Estimations des démonstrations manquantes

Temps estimés pour faire passer chaque exemple « Non prouvé » en « Démontré » : temps de code (écriture, test rapide, correction) et temps de calcul sur le poste local (RTX 5070 Ti Laptop, 12 Go). Mis à jour après chaque tâche terminée ; les résultats obtenus sont dans `exemples.yaml` et `session-status.md`.

Ordre de travail : du plus court en code au plus long.

| # | Exemple | Ce qu'il faut faire | Code | Calcul | Risque d'échec |
|---|---|---|---|---|---|
| 1 | SIGN04 préférences (DPO) | 3 graines : victoires 0,533 ± 0,032, gain net sur 2 graines sur 3. Reste : partir d'un modèle non aligné (base + SFT), ou s'appuyer sur le DPO itératif | 1 h | ~3 h | Moyen |
| 2 | SIGN04 variante DPO itératif | Gain net sur une graine : victoires 0,616 [0,573 ; 0,658] après 3 tours. 2 graines à lancer après SIGN07 | fait | ~3 h 20 | Faible à moyen |
| 3 | DATA01 connaissance interne | Démontré : extraction sur les personnages jamais questionnés 0,001 → 0,9996 sur 3 graines | fait | fait | — |
| 4 | DATA02 Whisper | Démontré sur EdAcc : WER 0,232 → 0,181 sur 3 graines | fait | fait | — |
| 5 | SIGN07 données synthétiques | Modèle de base amorcé, plafond ReST-EM : gain net sur une graine (0,363 → 0,402, témoin 0,332). 3 graines à lancer avec l'arrêt à la fin de la ligne « Answer: » | fait | ~6 à 9 h (3 graines) | Moyen |
| 5 bis | DATA01 variante encodeur | Protocole *Don't Stop Pretraining* sur banking77 : masquage continu de RoBERTa-base sur les messages sans étiquettes, classification avec 10 exemples par intention, témoin sans masquage, 3 graines | ~45 min, + 20 min de doc | ~30 min | Moyen : corpus du domaine petit (~150 000 tokens) ; repli : 5 exemples par intention ou plus de texte bancaire |
| 6 | FORM16 élagage | Fait à 0,5B : Wanda 0,66, 2:4 0,38, largeur 0,14, couches 0,09 (sur 0,88), ne paie pas à cette taille. Plans possibles sur le poste : voir « FORM16 : plans possibles » ci-dessous. À faire en dernier, plan à choisir | 2 à 3 h | 7 à 10 h | Moyen |

Total restant : calculs prêts à relancer (SIGN07 graines, DPO itératif graines), environ 9 h 30 ; FORM16 à part. Les calculs longs se lancent en file la nuit (scripts d'attente, voir `TODONEXT.md`).

## FORM16 : plans possibles sur le poste (12 Go, 30 Go de RAM)

L'élagage s'appuie sur la redondance des poids : à 50 % de parcimonie, Wanda et SparseGPT perdent peu à partir de 7B, beaucoup en dessous de 1B. Deux techniques lèvent la limite de mémoire de la cicatrisation :

- **sorties de l'enseignant calculées à l'avance** (32 tokens les plus probables par position, sur disque) : le modèle d'origine n'est plus en mémoire pendant l'entraînement ;
- **LoRA non fusionné** : la base élaguée reste creuse et gelée, l'adaptateur (rang 16, environ 1 % de calcul en plus) apprend par distillation. Le papier Wanda récupère ainsi la qualité après élagage. La fusion rendrait les poids denses.

| Plan | Contenu | Code | Calcul | Ce qu'il montre |
|---|---|---|---|---|
| A. Échelle de tailles | Wanda et 2:4, distillation avec LoRA non fusionné, même protocole à 0,5B, 1,5B et 3B | ~2 h | ~7 h (1 h, 2 h, 4 h) | Part de qualité conservée selon la taille : à partir de quelle taille l'élagage paie |
| B. 7B sans cicatrisation | Qwen2.5-7B-Instruct, Wanda et 2:4 couche par couche (modèle en RAM) ; référence et modèles élagués évalués en 8 bits (15 Go en bf16, ne tient pas sur la carte) | ~1 h de plus que A, ~2 h seul | ~3 h | L'affirmation du papier Wanda à la taille où elle a été faite ; mêle un peu élagage et quantification |
| C. 1,5B, cicatrisation complète | Élève en bf16 entraîné en entier (masque conservé) avec AdamW 8 bits, enseignant pré-calculé ; à la limite des 12 Go | ~1 h 30 | ~3 h (Wanda + 2:4) | Un 1,5B élagué à 50 % contre le 0,5B dense : élaguer un gros modèle ou prendre un petit |

Pour tous : mesurer la vitesse réelle du 2:4 avec les noyaux creux de PyTorch (gain non garanti), et 2 graines de plus pour le point qui conclut. Au-delà de 8B : 30 Go en bf16 pour un 14B, hors de portée du poste.
