# 04 — Optimisation des préférences (DPO et famille)

> Montrer au modèle des paires « réponse préférée / réponse rejetée » et le pousser vers la préférée, sans modèle de récompense ni boucle de RL.

| Regroupe | Modifie | Local 12 Go | Test `--quick` |
|---|---|---|---|
| DPO, DPO normalisé, IPO, rDPO, SLiC, APO, NCA, BCO, SPPO, EXO, AOT, DiscoPOP, ORPO, SimPO, CPO, KTO, DPO itératif / on-policy | Tous les poids (ou LoRA) | ✅ jusqu'à ~0,5B en complet (le modèle de référence double la mémoire), ~3B en LoRA | ✅ |

## Principe

Le RLHF classique entraîne d'abord un modèle de récompense, puis optimise la politique avec PPO ([05](../05-rl-en-ligne)). **DPO** montre que le même objectif (maximiser la préférence sans trop s'éloigner du modèle de référence) se ramène à une simple perte de classification sur les paires :

```
perte = -log σ( β · [ (log π(préférée) - log π_ref(préférée)) - (log π(rejetée) - log π_ref(rejetée)) ] )
```

Toutes les méthodes regroupées ici utilisent **les mêmes données** (prompt, préférée, rejetée) et la même boucle. Elles ne diffèrent que par la perte, d'où leur regroupement. On choisit avec `--set algo=...` :

| Algo | Idée | Référence nécessaire |
|---|---|---|
| `dpo` | Perte sigmoïde de DPO | oui |
| `dpo_norm` | DPO normalisé par la longueur (Tülu 3), limite la dérive vers les réponses longues | oui |
| `ipo` | Perte quadratique, moins de surapprentissage quand les préférences sont tranchées | oui |
| `rdpo` | Robuste aux paires mal étiquetées (`label_smoothing` = taux de bruit) | oui |
| `slic` | Perte hinge (SLiC-HF) | oui |
| `apo_zero` / `apo_down` | Ancrage : pousser la préférée vers le haut ou la rejetée vers le bas selon la qualité des données | oui |
| `nca`, `bco`, `sppo`, `exo`, `aot`, `discopop` | Autres formulations (probabiliste, classifieur, jeu, transport optimal, perte découverte par LLM) | oui |
| `orpo` | SFT + terme de ratio de cotes, en une seule étape | **non** |
| `simpo` | Récompense = log-prob moyenne, avec une marge γ | **non** |
| `cpo` | DPO sans référence + terme de vraisemblance (traduction) | **non** |
| `kto` | Pas de paires : chaque réponse est seulement « bonne » ou « mauvaise » (feedback 👍/👎) | oui |

Le **DPO itératif / on-policy** ([`variantes/dpo-iteratif`](variantes/dpo-iteratif)) fait générer les paires par le modèle lui-même à chaque tour. Un juge les classe : vérificateur, modèle de récompense ou LLM juge. Les paires restent ainsi dans la distribution du modèle, ce qui marche nettement mieux que des paires figées écrites par d'autres.

## Modèles concernés

| Type de modèle | Applicable | Remarques |
|---|---|---|
| LLM décodeur, idéalement après un SFT | ✅ cas principal | [`run.py`](run.py) |
| LLM instruct du commerce (Qwen-Instruct, Llama-Instruct...) | ✅ | Déjà SFT, on peut appliquer DPO directement |
| Modèle de base (sans SFT) | ⚠️ | Préférer ORPO, qui combine SFT et préférences, ou faire un SFT d'abord |
| Multimodal (Qwen2.5-VL, LLaVA, Idefics) | ✅ | `DPOTrainer` gère les images dans les prompts (réduction des hallucinations visuelles) |
| Diffusion (SD, SDXL) | ✅ | Diffusion-DPO : même idée appliquée à la perte de débruitage (scripts diffusers `train_diffusion_dpo`) |
| Encodeur (BERT) / classifieur | ❌ | Pas de génération, rien à préférer |
| Modèle fermé via API | ⚠️ | OpenAI propose un « preference fine-tuning » (DPO) sur ses modèles |

## Quand l'utiliser

- ✅ Après un SFT, pour aligner le ton, l'utilité, la sécurité, la concision.
- ✅ Vous avez ou pouvez produire des comparaisons : annotateurs, 👍/👎 des utilisateurs (KTO), LLM juge, vérificateur.
- ✅ Alternative bien plus simple et stable que PPO pour les préférences « subjectives ».
- ⚠️ Pour le raisonnement vérifiable (maths, code), le RL en ligne avec récompense vérifiable ([05](../05-rl-en-ligne)) va plus loin. Le DPO itératif est un intermédiaire.

## Utilisation

```bash
python run.py --quick
python run.py                                               # DPO complet Qwen2.5-0.5B-Instruct sur paires arithmétiques
python run.py --set algo=ipo --set beta=0.01
python run.py --set algo=simpo --set beta=2.0 --set entrainement.lr=1e-6
python run.py --set algo=kto
python run.py --set peft.activer=true --set entrainement.lr=5e-6 --set modele.nom=Qwen/Qwen2.5-3B-Instruct
python run.py --set donnees.source=hf --set donnees.nom=trl-lib/ultrafeedback_binarized --set donnees.split_train=train --set donnees.split_test=test
for a in dpo ipo rdpo slic apo_zero nca bco sppo orpo simpo cpo kto; do python run.py --quick --set algo=$a; done
python variantes/dpo-iteratif/run.py --quick
```

## Ce que fait l'exemple

- **Données** : pour chaque problème, `chosen` est le raisonnement juste et `rejected` le même raisonnement avec une erreur de calcul. C'est le cas le plus difficile à distinguer, et le plus utile.
- **Mesures** :
  - `exactitude_preferences` : part des paires test où le modèle préfère la bonne réponse (log-prob moyenne) ;
  - `marge_moyenne` ;
  - exactitude en génération.
- **Attendu** : `exactitude_preferences` monte nettement et la génération suit, plus modestement.

## Paramètres principaux (`config.yaml`)

| Clé | Rôle |
|---|---|
| `algo` | Voir le tableau |
| `beta` | Écart autorisé par rapport à la référence. Plus petit = le modèle s'éloigne davantage |
| `entrainement.lr` | **Très faible** en complet (5e-7). Un lr de SFT détruit le modèle |
| `peft.activer` | LoRA : la référence est la base sans adaptateur, ce qui évite de charger une deuxième copie |

## Grosse infra

- [`infra/config.yaml`](infra/config.yaml) : recette **Tülu 3** (DPO normalisé en longueur, β=5, mélange de 272k préférences) sur Llama-3.1-8B-SFT, DeepSpeed ZeRO-3 sur 8×H100. Pour 70B : 2 à 4 nœuds via Slurm.
- `precompute_ref_log_probs: true` (dans `entrainement.autres`) : les log-probs de la référence sont calculées une fois, puis la référence est déchargée.
- Frameworks : [open-instruct](https://github.com/allenai/open-instruct), [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF), LLaMA-Factory.
- Préférences on-policy à grande échelle : générer avec vLLM (n=8 à 16 par prompt), noter avec un modèle de récompense (Skywork-Reward) ou un LLM juge, puis DPO.

## Pièges

- **lr trop élevé** : c'est la première cause d'échec. Commencer à 5e-7 en complet.
- **Dérive de longueur** : DPO favorise les réponses longues. Utiliser `dpo_norm` ou `simpo`, surveiller la longueur moyenne.
- **Paires off-policy** : des paires écrites par un autre modèle apprennent moins. Préférer le DPO itératif.
- **« Déplacement de vraisemblance »** : la log-prob de la réponse préférée peut *baisser* pendant l'entraînement, seule la marge compte. Ajouter un peu de SFT (`loss_type: [sigmoid, sft]`) si ça dégrade la génération.
- **Chat template** : le même qu'au SFT.

## Références

- Rafailov et al., *DPO* (2023) · Azar et al., *IPO* (2023) · Ethayarajh et al., *KTO* (2024) · Hong et al., *ORPO* (2024) · Meng et al., *SimPO* (2024)
- Xu et al., *CPO* (2024) · D'Oosterlinck et al., *APO* (2024) · Chowdhury et al., *Robust DPO* (2024)
- Lambert et al., *Tülu 3* (2024) · Yuan et al., *Self-Rewarding Language Models* (2024) · Xiong et al., *Iterative Preference Learning* (2024)
