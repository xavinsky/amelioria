# 10 — Désapprentissage (machine unlearning)

> Faire **oublier** au modèle un ensemble précis de connaissances (données personnelles, contenu sous copyright, savoir dangereux), sans détruire le reste.

| Regroupe | Modifie | Local 12 Go | Test `--quick` |
|---|---|---|---|
| Gradient ascent, gradient difference, NPO (+ rétention), réponses « je ne sais pas » (IDK). En infra : SimNPO, RMU (open-unlearning) | Tous les poids | ✅ jusqu'à ~1B | ✅ |

## Principe

Schéma : [docs/schemas-methodes.md](../docs/schemas-methodes.md#m10) (ce que la méthode ajoute ou modifie dans le transformer).

On dispose d'un **ensemble à oublier** (forget set) et d'un **ensemble à retenir** (retain set). Toutes les méthodes suivent la même boucle : un lot à oublier, éventuellement un lot à retenir, puis une perte combinée. Seule la perte change (`--set desapprentissage.methode=...`) :

| Méthode | Perte sur l'oubli | Rétention | Comportement |
|---|---|---|---|
| `gradient_ascent` | `-NLL` : on *monte* la perte | non | Oublie vite, mais s'effondre vite (le modèle finit par tout désapprendre) |
| `gradient_difference` | `-NLL` | `+NLL` sur le retain set | Plus stable |
| `npo` | Perte de type DPO sans réponse préférée : `-(2/β)·log σ(-β·log(π/π_ref))` | optionnelle (`retenu: true`) | Bornée : l'oubli ralentit tout seul, pas d'effondrement. Référence actuelle |
| `idk` | SFT vers « je n'ai pas d'information » | `+NLL` | Refus explicite plutôt qu'oubli des poids |

Le protocole suit **TOFU** : des personnages fictifs sont d'abord appris (*implantation*), puis on fait oublier 10 % d'entre eux. Comme le modèle ne pouvait pas les connaître avant, on sait exactement ce qui doit disparaître. Cette phase d'implantation est incluse dans `run.py`.

Mesures :

| Mesure | Rôle |
|---|---|
| `oubli` (exactitude) et `nll_oubli` | Doivent respectivement baisser et monter |
| `retenu` et `nll_retenu` | Doivent rester stables |
| `perplexite_generale` | Non-régression sur les connaissances générales |

## Modèles concernés

| Type de modèle | Applicable | Remarques |
|---|---|---|
| LLM décodeur (toutes familles) | ✅ cas principal | [`run.py`](run.py) |
| LLM avec LoRA | ✅ | Désapprentissage sur adaptateur : réversible, moins destructeur |
| Classifieurs (vision, texte) | ✅ | Même idée (gradient ascent sur les exemples à oublier, SCRUB, fine-tuning sur le retain set) |
| Diffusion | ✅ très étudié | Oublier un style, un concept, une personne : ESD, concept ablation, UCE |
| Modèle fermé via API | ❌ | Seulement des filtres en entrée et en sortie |

## Quand l'utiliser

- ✅ Obligation de retrait d'une donnée (RGPD, droit à l'oubli), contenu sous licence retiré.
- ✅ Réduire des connaissances dangereuses (WMDP : RMU).
- ⚠️ Le désapprentissage approché n'offre **aucune garantie formelle** : l'information peut revenir par extraction ou par un léger fine-tuning. La seule garantie complète est de réentraîner sans la donnée.
- ⚠️ Si l'objectif est seulement de ne pas *dire* quelque chose : un filtre ou un refus (`idk`) peut suffire.

## Utilisation

```bash
python run.py --quick
python run.py                                                          # NPO + rétention, Qwen2.5-0.5B, 40 personnages, 10 % oubliés
python run.py --set desapprentissage.methode=gradient_ascent           # observer l'effondrement de `retenu`
python run.py --set desapprentissage.methode=gradient_difference
python run.py --set desapprentissage.methode=idk
python run.py --set donnees.part_oubli=0.05 --set entrainement.max_steps=120
python run.py --set donnees.source=tofu --set implantation.activer=false --set modele.nom=open-unlearning/tofu_Llama-3.2-1B-Instruct_full
```

## Ce que fait l'exemple

1. Génère 40 personnages (160 questions). Tire 4 personnages à oublier.
2. **Implantation** : SFT sur toutes les questions. Le modèle connaît alors les 40 personnages.
3. Mesure `oubli`, `retenu` et `perplexite_generale`.
4. 60 pas de NPO + rétention.
5. Remesure. Attendu : l'exactitude sur les 4 personnages oubliés chute, celle sur les 36 autres et la perplexité générale restent proches.

## Paramètres principaux (`config.yaml`)

| Clé | Rôle |
|---|---|
| `desapprentissage.methode` | Voir le tableau |
| `desapprentissage.beta` | NPO : plus grand = arrêt plus précoce |
| `desapprentissage.poids_retenu` | Équilibre entre oubli et rétention |
| `entrainement.max_steps`, `lr` | Trop de pas, et tout s'effondre (surtout en gradient ascent) |
| `donnees.source` | `faits` (généré) ou `tofu` (benchmark réel) |

## Grosse infra

- [`infra/config.yaml`](infra/config.yaml) : TOFU forget10 sur le modèle Llama-3.1-8B déjà fine-tuné sur TOFU (1 GPU 80 Go).
- [`infra/lancer.sh`](infra/lancer.sh) : [open-unlearning](https://github.com/locuslab/open-unlearning), le cadre de référence. Benchmarks TOFU, MUSE, WMDP ; méthodes GradAscent, GradDiff, NPO, SimNPO, DPO, RMU ; multi-GPU via accelerate/DeepSpeed ; métriques standard (forget quality, model utility, attaques par extraction).

## Pièges

- **Effondrement** : le gradient ascent sans rétention finit par produire du charabia. Toujours surveiller `retenu` et la perplexité générale.
- **Oubli superficiel** : la probabilité baisse sur la formulation exacte, mais une paraphrase ou quelques pas de fine-tuning font revenir l'information. Évaluer avec des paraphrases et des attaques.
- **Retain set mal choisi** : il doit contenir des connaissances *proches* de celles à oublier (même domaine), sinon on oublie aussi les voisins.

## Références

- Maini et al., *TOFU* (2024) · Zhang et al., *Negative Preference Optimization* (2024) · Fan et al., *SimNPO* (2024)
- Li et al., *WMDP / RMU* (2024) · Shi et al., *MUSE* (2024) · Eldan & Russinovich, *Who's Harry Potter?* (2023)
- Gandikota et al., *Erasing Concepts from Diffusion Models* (2023)
