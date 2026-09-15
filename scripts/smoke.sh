#!/usr/bin/env bash
# Lance le test rapide (--quick) de chaque méthode et de chaque variante, et résume les réussites/échecs.
#   bash scripts/smoke.sh                 # tout
#   bash scripts/smoke.sh 03 05           # seulement les dossiers dont le nom commence par 03 ou 05
#   CUDA_VISIBLE_DEVICES= bash scripts/smoke.sh   # forcer le CPU
set -uo pipefail
cd "$(dirname "$0")/.."
. .venv/bin/activate
mkdir -p logs/smoke

filtres=("$@")
scripts=$(find . -path ./.venv -prune -o -name run.py -print | grep -E '^\./[0-9]{2}-' | sort)
ok=(); ko=()
for s in $scripts; do
  dossier=$(dirname "$s" | sed 's|^\./||')
  if [ ${#filtres[@]} -gt 0 ] && ! printf '%s\n' "${filtres[@]}" | grep -q "^${dossier:0:2}"; then continue; fi
  journal="logs/smoke/$(echo "$dossier" | tr '/' '_').log"
  debut=$(date +%s)
  if python "$s" --quick > "$journal" 2>&1; then
    ok+=("$dossier ($(( $(date +%s) - debut ))s)")
  else
    ko+=("$dossier -> $journal")
  fi
  echo "[smoke] $dossier : $([ "${#ko[@]}" -gt 0 ] && [[ "${ko[-1]}" == "$dossier"* ]] && echo ECHEC || echo ok)"
done
echo; echo "Réussis (${#ok[@]}) :"; printf '  %s\n' "${ok[@]}"
echo "Échecs (${#ko[@]}) :"; printf '  %s\n' "${ko[@]}"
