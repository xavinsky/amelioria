# Choisir une méthode

## Par besoin

| Besoin | Commencer par | Si insuffisant |
|---|---|---|
| Le modèle ne connaît pas mes données (privées, récentes, volumineuses) | RAG ([EXEC18](../18-rag)) | Pré-entraînement continu ([DATA01](../01-pre-entrainement-continu)) + SFT |
| Format, ton, style ou tâche précise | Prompt ([EXEC17](../17-optimisation-prompt)) | LoRA ([DATA03](../03-peft)), puis SFT complet ([DATA02](../02-fine-tuning-supervise)) |
| Réponses « meilleures » selon un jugement humain | DPO ([SIGN04](../04-optimisation-preferences)) | RL avec modèle de récompense ([SIGN05](../05-rl-en-ligne)) |
| Raisonnement vérifiable (maths, code, SQL) | Calcul à l'inférence ([EXEC20](../20-calcul-inference)) | Rejection sampling ([SIGN07](../07-donnees-synthetiques)), puis RLVR ([SIGN05](../05-rl-en-ligne)) |
| Calcul exact, actions, données externes | Outils ([EXEC19](../19-outils-agents)) | SFT sur traces d'outils, RL multi-tours |
| Modèle plus petit ou moins cher, à qualité proche | Quantification ([FORM15](../15-quantification)) | Distillation ([SIGN06](../06-distillation)), élagage + distillation ([FORM16](../16-elagage)) |
| Plusieurs spécialisations dans un seul modèle | Fusion ([EDIT08](../08-fusion-modeles)) | Adaptateurs LoRA multiples servis ensemble ([DATA03](../03-peft)) |
| Corriger quelques faits précis | Édition ([EDIT09](../09-edition-connaissances)) | RAG ([EXEC18](../18-rag)) |
| Retirer des connaissances | Désapprentissage ([EDIT10](../10-desapprentissage)) | Filtre en entrée et en sortie ; réentraîner sans les données |
| Ajuster un trait (sentiment, formalité) sans données | Pilotage d'activations ([EDIT11](../11-pilotage-activations)) | DPO sur paires contrastées ([SIGN04](../04-optimisation-preferences)) |
| Textes plus longs que la fenêtre | RAG ([EXEC18](../18-rag)) | Extension de contexte ([FORM12](../12-extension-contexte)) |
| Langue ou domaine mal tokenisés | Extension de vocabulaire ([FORM13](../13-extension-vocabulaire)) + pré-entraînement continu | — |
| Plus de capacité sans pré-entraînement complet | Croissance d'architecture ([FORM14](../14-croissance-architecture)) | — |

## Par contrainte

| Contrainte | Méthodes accessibles |
|---|---|
| Modèle fermé (API) | 17, 18, 19, 20 ; fine-tuning « as a service » du fournisseur (SFT, DPO, RFT) |
| Aucun GPU | EXEC17 à EXEC20 via API ; fusion (EDIT08) ; GGUF (FORM15) ; édition et pilotage de petits modèles |
| Un GPU grand public (12 à 24 Go) | Tout, jusqu'à ~1 à 3B en entraînement complet, 8B en QLoRA |
| Pas de données annotées | 01 (texte brut), 07 (données générées + filtre), 05 RLVR (vérificateur), 11 (quelques paires), 17 à 20 |

## Recette type d'un modèle post-entraîné (Tülu 3, Llama 3, Qwen)

```
modèle de base
  -> (DATA01) pré-entraînement continu / mid-training sur données de qualité, long contexte (FORM12)
  -> (DATA02) SFT sur instructions, avec données synthétiques filtrées (SIGN07)
  -> (SIGN04) DPO sur préférences, dont préférences on-policy
  -> (SIGN05) RL à récompense vérifiable (maths, code, consignes)
  -> (EDIT08) fusion éventuelle de variantes
  -> (FORM15) quantification pour le service
  -> (17-20) prompt système, RAG, outils, calcul à l'inférence au déploiement
```
