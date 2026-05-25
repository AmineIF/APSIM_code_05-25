#!/bin/bash
#SBATCH --job-name=baseline_py
#SBATCH --output=/srv/lustre01/project/assiwat-tbwgr0oduuk/users/mohamed.benaly/outputsimulation/csvetfigures/baseline_py-%j.out
#SBATCH --error=/srv/lustre01/project/assiwat-tbwgr0oduuk/users/mohamed.benaly/outputsimulation/csvetfigures/baseline_py-%j.err
#SBATCH --time=05:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G

set -euo pipefail
set -x

source /srv/software/easybuild/software/Anaconda3/2024.06-1/bin/activate APENV

export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1

BASELINE_SCRIPT="/srv/lustre01/project/assiwat-tbwgr0oduuk/users/mohamed.benaly/inputsimulation/script/baseline_optimisation_with_optimum_csv.py"  
DB_SOURCE_DIR="/srv/lustre01/project/assiwat-tbwgr0oduuk/users/mohamed.benaly/outputsimulation/dbfiels"
OUT_DIR="/srv/lustre01/project/assiwat-tbwgr0oduuk/users/mohamed.benaly/outputsimulation/csvetfigures"

mkdir -p "$OUT_DIR"

if [[ ! -f "$BASELINE_SCRIPT" ]]; then
  echo "ERREUR: script baseline introuvable: $BASELINE_SCRIPT" >&2
  exit 1
fi

if [[ ! -d "$DB_SOURCE_DIR" ]]; then
  echo "ERREUR: dossier DB introuvable: $DB_SOURCE_DIR" >&2
  exit 1
fi

# Utiliser directement les DB sur le stockage source, sans copie locale.
export DB_DIR="$DB_SOURCE_DIR"
echo "DB_DIR exporté = $DB_DIR"

echo "===== DEMARRAGE BASELINE ====="
date
start_baseline=$(date +%s)

python "$BASELINE_SCRIPT"

end_baseline=$(date +%s)
echo "===== FIN BASELINE ====="
date
echo "DUREE BASELINE: $((end_baseline - start_baseline)) secondes"
