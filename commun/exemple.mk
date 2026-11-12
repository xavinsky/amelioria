# Inclus par le Makefile de chaque dossier de méthode : mêmes cibles que le Makefile racine, EX déduit du dossier.
#   cd 03-peft && make test [VAR=diffusion-lora] [SET="peft.methode=dora"]

EX := $(firstword $(subst -, ,$(notdir $(CURDIR))))
CIBLES := help liste verifier test run selecteurs smoke complet

.DEFAULT_GOAL := help
.PHONY: $(CIBLES)

$(CIBLES):
	@$(MAKE) --no-print-directory -C .. $@ EX=$(EX) VAR="$(VAR)" SET="$(SET)" API="$(API)" MODELE="$(MODELE)" LANCEUR="$(LANCEUR)"
