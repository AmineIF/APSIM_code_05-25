#!/bin/bash
#SBATCH --job-name=proj_min
#SBATCH --output=/srv/lustre01/project/assiwat-tbwgr0oduuk/users/mohamed.benaly/outputsimulation/csvetfigures/proj_min-%j.out
#SBATCH --error=/srv/lustre01/project/assiwat-tbwgr0oduuk/users/mohamed.benaly/outputsimulation/csvetfigures/proj_min-%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G

set -euo pipefail
set -x

source /srv/software/easybuild/software/Anaconda3/2024.06-1/bin/activate APENV

export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1

PROJECTION_SCRIPT="/srv/lustre01/project/assiwat-tbwgr0oduuk/users/mohamed.benaly/inputsimulation/script/projection_optimisation_with_optimum_csv_baseline_columns.py" 
DB_SOURCE_DIR="/srv/lustre01/project/assiwat-tbwgr0oduuk/users/mohamed.benaly/outputsimulation/dbfiels"
OUT_DIR="/srv/lustre01/project/assiwat-tbwgr0oduuk/users/mohamed.benaly/outputsimulation/csvetfigures"

mkdir -p "$OUT_DIR"

if [[ ! -f "$PROJECTION_SCRIPT" ]]; then
  echo "ERREUR: script projection introuvable: $PROJECTION_SCRIPT" >&2
  exit 1
fi

if [[ ! -d "$DB_SOURCE_DIR" ]]; then
  echo "ERREUR: dossier DB introuvable: $DB_SOURCE_DIR" >&2
  exit 1
fi

if [[ -n "${SLURM_TMPDIR:-}" ]]; then
  LOCAL_DB_DIR="$SLURM_TMPDIR/db_local_projection"
else
  LOCAL_DB_DIR="/tmp/$USER/db_local_projection_${SLURM_JOB_ID}"
fi
mkdir -p "$LOCAL_DB_DIR"

echo "===== COPIE DES DB EN LOCAL ====="
echo "Source DB: $DB_SOURCE_DIR"
echo "Local DB:  $LOCAL_DB_DIR"
copy_start=$(date +%s)
cp "$DB_SOURCE_DIR"/*.db "$LOCAL_DB_DIR"/
copy_end=$(date +%s)
echo "DUREE COPIE DB: $((copy_end - copy_start)) secondes"

export DB_DIR="$LOCAL_DB_DIR"
export OUT_DIR="$OUT_DIR"
echo "DB_DIR exporté = $DB_DIR"
echo "OUT_DIR exporté = $OUT_DIR"

echo "===== DEMARRAGE PROJECTION MINIMALE ====="
date
start_projection=$(date +%s)

python "$PROJECTION_SCRIPT"

end_projection=$(date +%s)
echo "===== FIN PROJECTION MINIMALE ====="
date
echo "DUREE PROJECTION: $((end_projection - start_projection)) secondes"
