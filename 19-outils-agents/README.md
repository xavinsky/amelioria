# 19 — Outils et boucle agentique (function calling)

> Donner au modèle des fonctions qu'il peut appeler (calculatrice, recherche, API, exécution de code) et une boucle qui exécute ces appels, puis lui renvoie les résultats jusqu'à la réponse finale.

| Regroupe | Modifie | Local 12 Go | Test `--quick` |
|---|---|---|---|
| Function / tool calling (chat template `tools`, API OpenAI), boucle agentique (ReAct), outils de calcul et de recherche. Voir aussi le dépôt voisin [`../agentique`](../../agentique) | Rien : l'environnement du modèle | ✅ (modèle 3B+) | ✅ |

## Principe

Schéma : [docs/schemas-methodes.md](../docs/schemas-methodes.md#m19) (ce que la méthode ajoute ou modifie dans le transformer).

Un LLM calcule mal de tête, ne connaît pas les données privées ni l'actualité, et ne peut pas agir. Avec des outils, il **délègue** :

```
[système + outils décrits en JSON Schema] + question
  -> modèle : <tool_call>{"name": "calculer", "arguments": {"expression": "347*29+1234"}}</tool_call>
  -> programme : exécute calculer(...) -> "11297"
  -> modèle (avec le résultat) : "... Answer: 11297"
```

- **Local** ([`run.py`](run.py), `boucle_locale`) : `tok.apply_chat_template(messages, tools=[fonctions])`. transformers génère le schéma JSON à partir des signatures et des docstrings, puis on analyse les balises `<tool_call>` (format Hermes, Qwen).
- **API** (`boucle_api`) : protocole OpenAI `tools` / `tool_calls`, compatible vLLM, Ollama et les fournisseurs cloud.

L'exemple pose deux types de questions **impossibles sans outil pour un petit modèle** :

1. un calcul à gros nombres, résolu avec `calculer` ;
2. une question sur un personnage fictif suivie d'un calcul (« combien d'années entre sa naissance et 2024 »), qui demande `chercher_fiche` puis `calculer`.

Mesures : exactitude sans outils, exactitude avec outils, nombre moyen d'appels.

## Modèles concernés

| Type de modèle | Applicable | Remarques |
|---|---|---|
| LLM instruct entraînés au tool calling (Qwen2.5/3, Llama 3.1+, Mistral, Hermes, GLM, DeepSeek) | ✅ cas principal | Le chat template doit accepter `tools` |
| LLM fermés via API (GPT, Claude, Gemini) | ✅ | `modele.api_base` |
| Petits modèles (< 3B) ou sans entraînement aux outils | ⚠️ | Appels mal formés. Solution : SFT sur des traces d'appels ([02](../02-fine-tuning-supervise), [03](../03-peft)) ou génération contrainte (JSON schema) |
| Multimodal | ✅ | Agents « computer use » : captures d'écran en entrée, actions en sortie |
| Encodeurs, diffusion | ❌ | Pas de boucle de décision. Ils peuvent en revanche *être* des outils |

## Quand l'utiliser

- ✅ Calcul exact, données à jour (recherche web, base SQL), actions (envoyer, créer, exécuter).
- ✅ Tâches en plusieurs étapes dont le modèle choisit l'enchaînement.
- ⚠️ Chaque tour relance le modèle, donc latence et coût se multiplient. Limiter `max_tours`.
- ⚠️ **Sécurité** : un outil qui exécute du code ou agit sur un système doit être isolé (bac à sable, liste blanche). L'outil `calculer` de l'exemple évalue un AST restreint, jamais `eval` brut.

## Utilisation

```bash
python run.py --quick
python run.py                                                         # Qwen2.5-3B-Instruct, 60 questions
python run.py --set modele.api_base=http://localhost:11434/v1 --set modele.nom=qwen3:8b     # via Ollama
python run.py --set max_tours=2
```

Pour aller plus loin en local (agents complets, Ollama, Hermes Agent, frameworks PydanticAI / smolagents / LangGraph) : le dépôt [`../agentique`](../../agentique).

## Paramètres principaux (`config.yaml`)

| Clé | Rôle |
|---|---|
| `modele.nom`, `api_base` | Modèle local ou servi |
| `systeme` / `systeme_sans_outils` | Consignes des deux conditions |
| `max_tours` | Nombre maximal d'aller-retours outil |
| `donnees.n_entites`, `evaluation.n` | Taille de la base et nombre de questions |

## Grosse infra

- [`infra/lancer.sh`](infra/lancer.sh) : Qwen2.5-72B servi par vLLM avec analyse native des appels (`--enable-auto-tool-choice --tool-call-parser hermes`), puis `run.py` en mode API ([`infra/config.yaml`](infra/config.yaml)).
- **Améliorer le modèle lui-même** pour les outils : SFT sur traces d'appels réussis, puis **RL multi-tours** avec récompense de réussite de la tâche. `GRPOTrainer(tools=..., environment_factory=...)` du dossier [05](../05-rl-en-ligne), verl, SkyRL.

## Pièges

- **Parser adapté à la famille** : le format des appels varie (Hermes, Llama JSON, Mistral).
- **Descriptions d'outils** : le modèle choisit d'après la docstring. Elle doit être précise et donner un exemple d'argument.
- **Boucles infinies** : borner les tours, détecter les appels répétés à l'identique.
- **Résultats d'outils non fiables** (web) : c'est un vecteur d'injection de prompt. Traiter leur contenu comme des données, jamais comme des instructions.

## Références

- Schick et al., *Toolformer* (2023) · Yao et al., *ReAct* (2022) · Patil et al., *Gorilla* (2023)
- Documentation transformers « Tool use » : https://huggingface.co/docs/transformers/chat_extras
- Dépôt voisin : [`../agentique`](../../agentique)
