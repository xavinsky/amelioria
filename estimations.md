# Estimations des démonstrations manquantes

Temps estimés pour faire passer chaque exemple « Non prouvé » en « Démontré » : temps de code (écriture, test rapide, correction) et temps de calcul sur le poste local (RTX 5070 Ti Laptop, 12 Go). Mis à jour après chaque tâche terminée ; les résultats obtenus sont dans `exemples.yaml` et `session-status.md`.

Ordre de travail : du plus court en code au plus long.

| # | Exemple | Ce qu'il faut faire | Code | Calcul | Risque d'échec |
|---|---|---|---|---|---|
| 1 | SIGN04 préférences (DPO) | Gain net sur une graine (16 000 paires UltraFeedback, lr 5e-5) : victoires 0,552 [0,508 ; 0,595]. Confirmer sur 2 graines (`make graines EX=SIGN04 N=3`), ou partir d'un modèle non aligné (base + SFT) | fait | ~4 h (2 graines de plus) | Moyen : gain faible |
| 2 | SIGN04 variante DPO itératif | Même jeu et même juge : à chaque tour, le modèle génère, le juge classe, un tour de DPO | 30 min après le n° 1 | ~2 h 30 (3 tours) | Moyen |
| 3 | DATA01 connaissance interne | Protocole *Physics of LM 3.1* : fiches de tous les personnages + questions-réponses d'une moitié, extraction mesurée sur l'autre moitié | 1 h 30 à 2 h | 1 à 2 h, réglages compris | Élevé : demande beaucoup de formulations par fait |
| 4 | DATA02 Whisper | Trouver un jeu audio public où whisper-small se trompe nettement (accents, vocabulaire spécialisé), puis l'adapter | 1 h à 1 h 30, recherche comprise | ~45 min | Moyen : dépend du jeu trouvé |
| 5 | SIGN07 données synthétiques | Partir de Qwen2.5-0.5B de base amorcé en few-shot ; plafond de solutions par question (ReST-EM) | 45 min | ~4 h | Moyen à élevé |
| 6 | FORM16 élagage | A : élagage structuré (largeur ou couches, accélération réelle), 2 à 3 réglages | 30 min | ~1 h | Élevé à 0,5B |
| | | B : conclusion documentée « ne paie pas à cette taille » | 15 min | aucun | |

Total restant : environ 6 à 8 h de code et 10 à 13 h de calcul, graines comprises. Les calculs longs se lancent en file la nuit (scripts d'attente, voir `TODONEXT.md`).
