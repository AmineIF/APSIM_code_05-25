#!/bin/bash
#SBATCH --job-name=apsim_run
#SBATCH --output=/srv/lustre01/project/assiwat-tbwgr0oduuk/users/mohamed.benaly/outputsimulation/dbfiels/slurm-%j.out
#SBATCH --error=/srv/lustre01/project/assiwat-tbwgr0oduuk/users/mohamed.benaly/outputsimulation/dbfiels/slurm-%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G

set -euo pipefail
set -x

SIM_DIR="/srv/lustre01/project/assiwat-tbwgr0oduuk/users/mohamed.benaly/inputsimulation/fichierapsimx"
MET_DIR="/srv/lustre01/project/assiwat-tbwgr0oduuk/users/mohamed.benaly/inputsimulation/fichier.met"
CO2_DIR="/srv/lustre01/project/assiwat-tbwgr0oduuk/users/mohamed.benaly/inputsimulation/fichierco2"
OUT_DIR="/srv/lustre01/project/assiwat-tbwgr0oduuk/users/mohamed.benaly/outputsimulation/dbfiels"
SIF="$HOME/apsimng_latest.sif"

# Séquentiel d'abord, puis tu pourras tester 2 plus tard
MAX_JOBS="${MAX_JOBS:-1}"

# Très utile pour les relances
SKIP_EXISTING_DB="${SKIP_EXISTING_DB:-1}"

echo "===== DEMARRAGE JOB APSIM ====="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "User: $(whoami)"
echo "MAX_JOBS=$MAX_JOBS"
echo "SKIP_EXISTING_DB=$SKIP_EXISTING_DB"

mkdir -p "$OUT_DIR"

if [[ ! -d "$SIM_DIR" ]]; then
  echo "ERREUR: dossier SIM_DIR introuvable: $SIM_DIR" >&2
  exit 1
fi

if [[ ! -d "$MET_DIR" ]]; then
  echo "ERREUR: dossier MET_DIR introuvable: $MET_DIR" >&2
  exit 1
fi

if [[ ! -d "$CO2_DIR" ]]; then
  echo "ERREUR: dossier CO2_DIR introuvable: $CO2_DIR" >&2
  exit 1
fi

if [[ ! -f "$SIF" ]]; then
  echo "ERREUR: image Apptainer introuvable: $SIF" >&2
  exit 1
fi

if ! command -v apptainer >/dev/null 2>&1; then
  echo "ERREUR: commande apptainer introuvable dans l'environnement." >&2
  exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "ERREUR: commande python3 introuvable dans l'environnement." >&2
  exit 1
fi

# TMP/CACHE local Apptainer
if [[ -n "${SLURM_TMPDIR:-}" ]]; then
  export APPTAINER_TMPDIR="$SLURM_TMPDIR/apptainer_tmp"
  export APPTAINER_CACHEDIR="$SLURM_TMPDIR/apptainer_cache"
  WORK_DIR="$SLURM_TMPDIR/apsim_work"
  mkdir -p "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR" "$WORK_DIR"

  LOCAL_SIF="$SLURM_TMPDIR/apsimng_latest.sif"
  cp "$SIF" "$LOCAL_SIF"
else
  export APPTAINER_TMPDIR="/tmp/$USER/apptainer_tmp"
  export APPTAINER_CACHEDIR="/tmp/$USER/apptainer_cache"
  WORK_DIR="/tmp/$USER/apsim_work_${SLURM_JOB_ID}"
  mkdir -p "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR" "$WORK_DIR"

  LOCAL_SIF="$SIF"
fi

echo "APPTAINER_TMPDIR=$APPTAINER_TMPDIR"
echo "APPTAINER_CACHEDIR=$APPTAINER_CACHEDIR"
echo "WORK_DIR=$WORK_DIR"
echo "LOCAL_SIF=$LOCAL_SIF"

apptainer exec "$LOCAL_SIF" true

shopt -s nullglob
APSIM_FILES=("$SIM_DIR"/*.apsimx)

if (( ${#APSIM_FILES[@]} == 0 )); then
  echo "ERREUR: aucun fichier .apsimx trouvé dans $SIM_DIR" >&2
  exit 1
fi

if ! [[ "$MAX_JOBS" =~ ^[0-9]+$ ]] || (( MAX_JOBS < 1 )); then
  echo "ERREUR: MAX_JOBS doit être un entier >= 1. Valeur reçue: $MAX_JOBS" >&2
  exit 1
fi

echo "Nombre de fichiers APSIMX trouvés: ${#APSIM_FILES[@]}"

declare -a PIDS=()
declare -a PID_NAMES=()
FAIL=0

run_one_sim() {
  local SIM_SRC="$1"
  local SIM_FILE
  local BASE_NAME
  local SIM_PATH_OUT
  local DB_FILE
  local LOG_FILE
  local FINAL_DB_FILE
  local FINAL_LOG_FILE
  local start_one
  local end_one

  SIM_FILE="$(basename "$SIM_SRC")"
  BASE_NAME="${SIM_FILE%.apsimx}"

  SIM_PATH_OUT="$WORK_DIR/$SIM_FILE"
  DB_FILE="$WORK_DIR/${BASE_NAME}.db"
  LOG_FILE="$WORK_DIR/${BASE_NAME}.log"

  FINAL_DB_FILE="$OUT_DIR/${BASE_NAME}.db"
  FINAL_LOG_FILE="$OUT_DIR/${BASE_NAME}.log"

  echo "========================================"
  echo "Traitement de: $SIM_FILE"
  echo "Source: $SIM_SRC"
  echo "Copie de travail locale: $SIM_PATH_OUT"
  echo "DB locale attendue: $DB_FILE"
  echo "DB finale: $FINAL_DB_FILE"
  echo "Log local: $LOG_FILE"
  echo "Log final: $FINAL_LOG_FILE"
  echo "========================================"

  if [[ "$SKIP_EXISTING_DB" == "1" && -s "$FINAL_DB_FILE" ]]; then
    echo "DB déjà présente, on saute: $FINAL_DB_FILE"
    return 0
  fi

  rm -f "$SIM_PATH_OUT" "$DB_FILE" "$LOG_FILE"
  cp "$SIM_SRC" "$SIM_PATH_OUT"

  python3 - "$SIM_PATH_OUT" <<'PY'
from pathlib import Path
import re
import sys

p = Path(sys.argv[1])
txt = p.read_text(encoding="utf-8")

def fix_path(old: str) -> str:
    norm = re.sub(r'\\+', '/', old)

    if norm.lower().endswith('.met'):
        marker = "/data/cmip6/"
        if marker in norm:
            rel = norm.split(marker, 1)[1].lstrip("/")
            return f"/met/{rel}"

    if norm.lower().endswith('.csv') and 'co2_' in norm.lower():
        base = norm.split("/")[-1]
        return f"/opt/Examples/WeatherFiles/{base}"

    return old

def repl(m):
    old = m.group(2)
    new = fix_path(old)
    return m.group(1) + new + m.group(3)

new_txt = re.sub(r'("FileName"\s*:\s*")([^"]+)(")', repl, txt)
p.write_text(new_txt, encoding="utf-8")
print(f"Correction terminée pour {p.name}")
PY

  start_one=$(date +%s)
  echo "DEBUT $SIM_FILE : $(date)"

  apptainer run \
    --bind "$WORK_DIR:/work" \
    --bind "$MET_DIR:/met" \
    --bind "$CO2_DIR:/opt/Examples/WeatherFiles" \
    --pwd /work \
    "$LOCAL_SIF" \
    "/work/$SIM_FILE" \
    > "$LOG_FILE" 2>&1

  end_one=$(date +%s)
  echo "FIN $SIM_FILE : $(date)"
  echo "DUREE $SIM_FILE : $((end_one - start_one)) secondes"

  if [[ -s "$DB_FILE" ]]; then
    cp "$DB_FILE" "$FINAL_DB_FILE"
    cp "$LOG_FILE" "$FINAL_LOG_FILE"
    echo "OK: DB générée: $FINAL_DB_FILE"
  else
    echo "ERREUR: aucune DB valide trouvée après exécution pour $SIM_FILE" >&2
    echo "Consulte le log local: $LOG_FILE" >&2
    return 1
  fi

  echo "Terminé: $SIM_FILE"
}

wait_batch() {
  local i
  for i in "${!PIDS[@]}"; do
    if ! wait "${PIDS[$i]}"; then
      echo "ECHEC: ${PID_NAMES[$i]}" >&2
      FAIL=1
    fi
  done
  PIDS=()
  PID_NAMES=()
}

for SIM_SRC in "${APSIM_FILES[@]}"; do
  run_one_sim "$SIM_SRC" &
  PIDS+=("$!")
  PID_NAMES+=("$(basename "$SIM_SRC")")

  if (( ${#PIDS[@]} >= MAX_JOBS )); then
    wait_batch
  fi
done

if (( ${#PIDS[@]} > 0 )); then
  wait_batch
fi

if (( FAIL != 0 )); then
  echo "===== FIN JOB APSIM AVEC ERREURS =====" >&2
  exit 1
fi

SUCCESS_COUNT=$(find "$OUT_DIR" -maxdepth 1 -type f -name "*.db" | wc -l)

echo "===== FIN JOB APSIM ====="
echo "Nombre total de DB générées dans $OUT_DIR: $SUCCESS_COUNT"
date