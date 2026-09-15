# Choisir une méthode

## Par besoin

| Besoin | Commencer par | Si insuffisant |
|---|---|---|
| Le modèle ne connaît pas mes données (privées, récentes, volumineuses) | RAG ([18](../18-rag)) | Pré-entraînement continu ([01](../01-pre-entrainement-continu)) + SFT |
| Format, ton, style ou tâche précise | Prompt ([17](../17-optimisation-prompt)) | LoRA ([03](../03-peft)), puis SFT complet ([02](../02-fine-tuning-supervise)) |
| Réponses « meilleures » selon un jugement humain | DPO ([04](../04-optimisation-preferences)) | RL avec modèle de récompense ([05](../05-rl-en-ligne)) |
| Raisonnement vérifiable (maths, code, SQL) | Calcul à l'inférence ([20](../20-calcul-inference)) | Rejection sampling ([07](../07-donnees-synthetiques)), puis RLVR ([05](../05-rl-en-ligne)) |
| Calcul exact, actions, données externes | Outils ([19](../19-outils-agents)) | SFT sur traces d'outils, RL multi-tours |
| Modèle plus petit ou moins cher, à qualité proche | Quantification ([15](../15-quantification)) | Distillation ([06](../06-distillation)), élagage + distillation ([16](../16-elagage)) |
| Plusieurs spécialisations dans un seul modèle | Fusion ([08](../08-fusion-modeles)) | Adaptateurs LoRA multiples servis ensemble ([03](../03-peft)) |
| Corriger quelques faits précis | Édition ([09](../09-edition-connaissances)) | RAG ([18](../18-rag)) |
| Retirer des connaissances | Désapprentissage ([10](../10-desapprentissage)) | Filtre en entrée et en sortie ; réentraîner sans les données |
| Ajuster un trait (sentiment, formalité) sans données | Pilotage d'activations ([11](../11-pilotage-activations)) | DPO sur paires contrastées ([04](../04-optimisation-preferences)) |
| Textes plus longs que la fenêtre | RAG ([18](../18-rag)) | Extension de contexte ([12](../12-extension-contexte)) |
| Langue ou domaine mal tokenisés | Extension de vocabulaire ([13](../13-extension-vocabulaire)) + pré-entraînement continu | — |
| Plus de capacité sans pré-entraînement complet | Croissance d'architecture ([14](../14-croissance-architecture)) | — |

## Par contrainte

| Contrainte | Méthodes accessibles |
|---|---|
| Modèle fermé (API) | 17, 18, 19, 20 ; fine-tuning « as a service » du fournisseur (SFT, DPO, RFT) |
| Aucun GPU | 17 à 20 via API ; fusion (08) ; GGUF (15) ; édition et pilotage de petits modèles |
| Un GPU grand public (12 à 24 Go) | Tout, jusqu'à ~1 à 3B en entraînement complet, 8B en QLoRA |
| Pas de données annotées | 01 (texte brut), 07 (données générées + filtre), 05 RLVR (vérificateur), 11 (quelques paires), 17 à 20 |

## Recette type d'un modèle post-entraîné (Tülu 3, Llama 3, Qwen)

```
modèle de base
  -> (01) pré-entraînement continu / mid-training sur données de qualité, long contexte (12)
  -> (02) SFT sur instructions, avec données synthétiques filtrées (07)
  -> (04) DPO sur préférences, dont préférences on-policy
  -> (05) RL à récompense vérifiable (maths, code, consignes)
  -> (08) fusion éventuelle de variantes
  -> (15) quantification pour le service
  -> (17-20) prompt système, RAG, outils, calcul à l'inférence au déploiement
```
