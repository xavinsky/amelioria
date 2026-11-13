# Oubli catastrophique

Quand on entraîne un modèle sur un domaine ou une tâche étroits, il **perd des capacités qu'il avait** : connaissances générales, format de chat, langues, sécurité. Plus le nouvel entraînement est long, étroit et à lr élevé, plus la perte est forte.

## Le mesurer

Toujours suivre, avant et après, au moins une métrique **hors de la tâche visée** :

- exactitude sur une tâche hors sujet que le modèle de départ réussit : `evaluation.non_regression` (arithmétique en [DATA02](../02-fine-tuning-supervise) et [DATA03](../03-peft)) ;
- perplexité sur du texte général : `evaluation.perplexite_generale: true` dans les configs, sur wikitext ;
- quelques benchmarks généraux : `commun/infra/evaluer_lm_eval.sh` (MMLU, HellaSwag, IFEval) ;
- exactitude sur les connaissances voisines à préserver (`retenu` en [EDIT10](../10-desapprentissage), `specificite` en [EDIT09](../09-edition-connaissances)).

## Ce qu'on mesure dans le dépôt

Routage de messages bancaires vers 77 intentions (banking77), Qwen2.5-0.5B-Instruct ; non-régression : 200 problèmes d'arithmétique que le modèle réussit à 0,955 en raisonnant pas à pas.

| Entraînement | Intentions | Non-régression |
|---|---|---|
| Fine-tuning complet, sans remélange | 0,932 | 0,025 |
| Fine-tuning complet, 15 % de remélange | 0,904 | 0,765 |
| LoRA, sans remélange | 0,925 | 0,33 |
| LoRA, 15 % de remélange | 0,891 | 0,95 |

Entraîné à répondre par un mot, le modèle répond par un mot partout et cesse de raisonner. Le remélange limite cet oubli, LoRA davantage encore, et les deux ensemble le rendent non mesurable.

## Le limiter

| Technique | Principe | Où dans le dépôt |
|---|---|---|
| **Replay** | Mélanger 5 à 30 % de données générales aux données du domaine | `donnees.replay_ratio` en [DATA01](../01-pre-entrainement-continu) ; `donnees.remelange` en [DATA02](../02-fine-tuning-supervise) et [DATA03](../03-peft), avec les réponses du modèle d'origine |
| **LR faible et entraînement court** | Moins de pas, moins de dérive | Toutes les configs `entrainement.lr` |
| **PEFT (LoRA)** | Les poids d'origine ne bougent pas ; « LoRA learns less and forgets less » | [DATA03](../03-peft) |
| **Régularisation vers le modèle d'origine** | Pénalité KL (RL, DPO) ou distance aux poids (EWC, L2-SP) | `beta` en [SIGN04](../04-optimisation-preferences) et [SIGN05](../05-rl-en-ligne) |
| **Fusion avec l'original** | Interpoler le modèle spécialisé avec l'original (WiSE-FT) | [EDIT08](../08-fusion-modeles), méthode `lineaire` ou `slerp` |
| **Expansion de blocs** | Ajouter des couches entraînables et geler les couches d'origine | `expansion_blocs` en [FORM14](../14-croissance-architecture) |
| **Ne pas toucher aux poids** | RAG, outils, prompts | [EXEC17](../17-optimisation-prompt) à [EXEC20](../20-calcul-inference) |

## EWC en quelques lignes

Elastic Weight Consolidation pénalise l'écart aux poids d'origine, pondéré par l'importance de chaque poids (information de Fisher diagonale estimée sur des données générales) :

```python
# avant l'entraînement : importance F_i = E[(∂ log p / ∂θ_i)²] sur des données générales
fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
for lot in lots_generaux:
    model.zero_grad()
    model(**lot, labels=lot["input_ids"]).loss.backward()
    for n, p in model.named_parameters():
        fisher[n] += p.grad.detach() ** 2 / len(lots_generaux)
theta0 = {n: p.detach().clone() for n, p in model.named_parameters()}

# pendant l'entraînement : perte = perte_tache + λ/2 · Σ F_i (θ_i - θ0_i)²
penalite = sum((fisher[n] * (p - theta0[n]) ** 2).sum() for n, p in model.named_parameters())
perte = perte_tache + lam / 2 * penalite
```

Coût mémoire : deux copies supplémentaires des poids. Pour les LLM, le replay et LoRA sont en pratique plus simples et plus efficaces.
