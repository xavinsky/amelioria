#!/usr/bin/env bash
# Lance le test rapide (--quick) pour chaque valeur des sélecteurs d'algorithme, et résume les réussites/échecs.
#   bash scripts/selecteurs.sh              # tout
#   bash scripts/selecteurs.sh 04 15        # seulement les dossiers dont le nom commence par 04 ou 15
set -uo pipefail
cd "$(dirname "$0")/.."
. .venv/bin/activate
mkdir -p logs/selecteurs

# dossier|clé|valeurs|surcharges supplémentaires
CAS=(
  "03-peft|peft.methode|lora qlora dora rslora pissa loha lokr ia3 prompt_tuning prefix_tuning|"
  "04-optimisation-preferences|algo|$(python -c 'import sys; sys.path.insert(0, "04-optimisation-preferences"); import preferences; print(" ".join(preferences.ALGOS))')|"
  "05-rl-en-ligne|algo|grpo dr_grpo dapo gspo cispo rloo|"
  "05-rl-en-ligne|recompense.source|modele juge|--set recompense.juge=HuggingFaceTB/SmolLM2-135M-Instruct"
  "06-distillation|mode|sequence logits on_policy|"
  "07-donnees-synthetiques|strategie|rejet self_instruct critique_revision|"
  "08-fusion-modeles|methode|lineaire slerp arithmetique_taches ties dare_ties dare_lineaire|"
  "09-edition-connaissances|edition.methode|rome ft_contraint|"
  "10-desapprentissage|desapprentissage.methode|gradient_ascent gradient_difference npo idk|"
  "11-pilotage-activations|pilotage.mode|vecteur ablation|"
  "12-extension-contexte|contexte.methode|linear dynamic yarn abf|"
  "12-extension-contexte|contexte.methode|dynamic|--set entrainement.max_steps=0"
  "14-croissance-architecture|croissance.methode|profondeur expansion_blocs moe|"
  "15-quantification|methode|bnb_nf4 bnb_int8 torchao_int8wo torchao_int4wo torchao_fp8 qat|"
  "16-elagage|methode|magnitude wanda 2:4 couches largeur|"
  "16-elagage|cicatrisation.mode|aucune lm distillation|"
)

filtres=("$@")
ok=(); ko=()
for cas in "${CAS[@]}"; do
  IFS='|' read -r dossier cle valeurs extra <<< "$cas"
  if [ ${#filtres[@]} -gt 0 ] && ! printf '%s\n' "${filtres[@]}" | grep -q "^${dossier:0:2}"; then continue; fi
  for v in $valeurs; do
    nom="$dossier $cle=$v${extra:+ $extra}"
    journal="logs/selecteurs/${dossier}_${cle}=${v//:/-}${extra:+_extra}.log"
    debut=$(date +%s)
    # shellcheck disable=SC2086
    if python "$dossier/run.py" --quick --set "$cle=$v" $extra --sortie "sorties/selecteurs/${dossier}_${cle}=${v//:/-}" > "$journal" 2>&1; then
      ok+=("$nom ($(( $(date +%s) - debut ))s)"); echo "[selecteurs] $nom : ok"
    else
      ko+=("$nom -> $journal"); echo "[selecteurs] $nom : ECHEC"
    fi
  done
done
echo; echo "Réussis (${#ok[@]}) :"; printf '  %s\n' "${ok[@]}"
echo "Échecs (${#ko[@]}) :"; printf '  %s\n' "${ko[@]}"
