# AmeliorIA — installer et lancer n'importe quel exemple du dépôt.
#
#   make install [EXTRAS=1] [CUDA=cu130]    venv + socle (+ extras : bitsandbytes, torchao, diffusers, torchcodec)
#   make liste                              exemples par catégorie, avec statut et résultat mesuré
#   make test EX=DATA03 [VAR=diffusion-lora]    test rapide (--quick) : vérifie que tout s'exécute
#   make run  EX=DATA03 [VAR=...] [SET="peft.methode=dora entrainement.lr=1e-4"]
#   make run  EX=EXEC18 API=http://localhost:11434/v1 MODELE=qwen3:8b    modèle servi par une API compatible OpenAI
#
# EX : identifiant de la méthode (code de catégorie + numéro, voir make liste) ; le numéro seul (EX=03) marche aussi.
#
# Dans un dossier de méthode, les mêmes cibles marchent sans EX : cd 03-peft && make test

.DEFAULT_GOAL := help
SHELL := bash

EX ?=
VAR ?=
SET ?=
API ?=
MODELE ?=
CUDA ?=
EXTRAS ?=
# Préfixe de lancement facultatif (ex. LANCEUR="nice -n 10" ou un gestionnaire de file GPU).
LANCEUR ?=

VENV := .venv
PY := $(VENV)/bin/python
PY_GPTQ := .venv-llmcompressor/bin/python
LLAMACPP ?= $(HOME)/llama.cpp

# Numéro de dossier (« 03 ») de l'exemple désigné par EX, pour les scripts qui filtrent les dossiers.
NUM_EX = $(if $(EX),$(shell python3 scripts/catalogue.py chemin "$(EX)" | cut -c1-2))
# GPTQ et AWQ tournent dans leur propre venv (llm-compressor impose ses versions de torch et transformers).
PY_EXEMPLE := $(if $(filter methode=gptq methode=awq,$(SET)),$(PY_GPTQ),$(PY))
ARGS_SET := $(foreach s,$(SET),--set $(s)) $(if $(API),--set modele.api_base=$(API)) $(if $(MODELE),--set modele.nom=$(MODELE))
ARGS_VERIF := $(foreach s,$(SET),--set $(s)) $(if $(API),--api $(API))

.PHONY: help install install-gptq install-llamacpp liste readme site verifier test run selecteurs smoke complet

help: ## cette aide
	@echo "AmeliorIA — améliorer un modèle d'IA sans le réentraîner de zéro"
	@echo
	@grep -hE '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  make %-17s %s\n", $$1, $$2}'
	@echo
	@echo "Options : EX=DATA03 (ou 03)  VAR=variante  SET=\"cle=val ...\"  API=url  MODELE=nom  EXTRAS=1  CUDA=cu130  LANCEUR=..."
	@echo "Exemple : make install EXTRAS=1 && make liste && make test EX=FORM15"

install: ## crée .venv et installe le socle (EXTRAS=1 : extras ; CUDA=cu130 : index PyTorch)
	python3 -m venv $(VENV)
	$(PY) -m pip install -U pip
	$(if $(CUDA),$(PY) -m pip install torch torchvision --index-url https://download.pytorch.org/whl/$(CUDA))
	$(PY) -m pip install -r requirements.txt
	$(if $(EXTRAS),grep -v '^#' requirements-extras.txt | sed 's/#.*//' | grep -vE 'llmcompressor|deepspeed' | xargs $(PY) -m pip install)
	@echo "Installé. Suite : make liste, puis make test EX=<identifiant>"

install-gptq: ## venv séparé .venv-llmcompressor pour GPTQ / AWQ (FORM15)
	python3 -m venv .venv-llmcompressor
	$(PY_GPTQ) -m pip install -U pip
	$(PY_GPTQ) -m pip install llmcompressor datasets pyyaml

install-llamacpp: ## clone et compile llama.cpp dans LLAMACPP (défaut ~/llama.cpp) pour FORM15 VAR=gguf-llamacpp
	@[ -d $(LLAMACPP) ] || git clone --depth 1 https://github.com/ggml-org/llama.cpp $(LLAMACPP)
	@command -v cmake >/dev/null || $(PY) -m pip install cmake
	PATH=$(VENV)/bin:$$PATH cmake -S $(LLAMACPP) -B $(LLAMACPP)/build $(if $(shell command -v nvcc),-DGGML_CUDA=ON)
	PATH=$(VENV)/bin:$$PATH cmake --build $(LLAMACPP)/build -j --config Release
	$(PY) -m pip install $(LLAMACPP)/gguf-py

liste: ## les exemples par catégorie, avec statut et résultat mesuré
	@python3 scripts/catalogue.py liste

readme: ## régénère la section des méthodes du README depuis exemples.yaml
	@python3 scripts/catalogue.py readme

site: ## construit le site AmeliorIA (site/www/), à ouvrir dans un navigateur
	python3 site/build/build.py
	@echo "Ouvrir site/www/index.html"

$(PY):
	@echo "Environnement absent : lancer d'abord make install (EXTRAS=1 pour les extras)"; exit 1

verifier: | $(PY) ## vérifie les prérequis d'un exemple (EX, VAR, SET, API)
	@$(PY) scripts/catalogue.py verifier "$(EX)" "$(VAR)" $(ARGS_VERIF) $(if $(COMPLET),--complet)

test: | $(PY) ## test rapide d'un exemple (--quick)
	@$(PY) scripts/catalogue.py verifier "$(EX)" "$(VAR)" $(ARGS_VERIF)
	@d=$$(python3 scripts/catalogue.py chemin "$(EX)" "$(VAR)") && set -x && \
	  $(LANCEUR) $(PY_EXEMPLE) $$d/run.py --quick $(ARGS_SET)

run: | $(PY) ## exemple complet (config.yaml), résultats dans sorties/
	@$(PY) scripts/catalogue.py verifier "$(EX)" "$(VAR)" $(ARGS_VERIF) --complet
	@d=$$(python3 scripts/catalogue.py chemin "$(EX)" "$(VAR)") && set -x && \
	  $(LANCEUR) $(PY_EXEMPLE) $$d/run.py $(ARGS_SET)

selecteurs: | $(PY) ## chaque valeur des sélecteurs (méthode, algo...) en --quick (EX facultatif)
	$(LANCEUR) bash scripts/selecteurs.sh $(NUM_EX)

smoke: | $(PY) ## test rapide de tous les exemples (EX pour un seul dossier), résumé et journaux dans logs/smoke/
	$(LANCEUR) bash scripts/smoke.sh $(NUM_EX)

complet: | $(PY) ## exemples complets de tous les dossiers (ou EX), journaux dans logs/complet/
	COMPLET=1 $(LANCEUR) bash scripts/smoke.sh $(NUM_EX)
