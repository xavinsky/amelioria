---
title: Installer et lancer un exemple
id: installer
---

Chaque méthode a son dossier dans le dépôt, avec un exemple exécutable sur un poste équipé d'un GPU de 12 Go. Toutes les commandes passent par un `Makefile` à la racine : la même commande lance n'importe quel exemple, et vérifie d'abord ce qu'il lui faut.

## Installer

Python 3.13 et un GPU NVIDIA sont recommandés ; les méthodes d'inférence (prompt, RAG, outils, calcul à l'inférence) tournent aussi sans GPU, avec un modèle servi par API.

```bash
git clone <url du dépôt> ameliorer-modele-ia && cd ameliorer-modele-ia
make install EXTRAS=1 CUDA=cu130   # venv .venv, socle et extras (bitsandbytes, torchao, diffusers, torchcodec)
make liste                         # les exemples par catégorie, avec statut et résultat mesuré
```

`CUDA` choisit l'index PyTorch de la version CUDA du poste (`cu130`, `cu128`...) ; sans lui, PyTorch vient de PyPI. Deux installations servent à des exemples précis :

- `make install-gptq` : environnement séparé pour la quantification GPTQ et AWQ, dont la bibliothèque impose ses propres versions de PyTorch et de transformers ;
- `make install-llamacpp` : clone et compile llama.cpp pour l'export GGUF.

## Lancer

```bash
make test EX=FORM15                 # test rapide : petit modèle, quelques pas, quelques minutes au plus
make run  EX=FORM15                 # exemple complet, résultats dans sorties/
make test EX=DATA03 VAR=diffusion-lora # une variante
make run  EX=DATA03 SET="peft.methode=dora entrainement.lr=1e-4"
make run  EX=EXEC18 API=http://localhost:11434/v1 MODELE=qwen3:8b
```

| Option | Rôle |
|---|---|
| `EX` | Méthode : identifiant (`DATA03`, code de catégorie + numéro), numéro seul (`03`) ou nom de dossier (`peft`) |
| `VAR` | Variante du dossier |
| `SET` | Surcharges de la configuration, séparées par des espaces |
| `API`, `MODELE` | Modèle servi par une API compatible OpenAI (Ollama, vLLM, fournisseur cloud) |

Avant de lancer quoi que ce soit, `make test` et `make run` vérifient les modules, le GPU et sa mémoire, llama.cpp et l'API. S'il manque quelque chose, ils disent quoi installer. `make verifier EX=FORM15` ne fait que cette vérification. Dans un dossier de méthode, les mêmes cibles marchent sans `EX` : `cd 03-peft && make test`.

## Ce que produit un exemple

Chaque exemple mesure le modèle avant, applique la méthode, puis remesure dans les mêmes conditions. Il affiche un tableau avant/après et écrit `sorties/<dossier>/resultats.json`, avec le modèle ou l'adaptateur produit. Le statut de chaque exemple et ses résultats de référence sont dans [Statut des exemples](#statut-des-exemples).
