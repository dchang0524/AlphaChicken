#!/bin/bash
#SBATCH --job-name=MagnusRL              # Job name
#SBATCH -N1 --ntasks-per-node=1          # One node, single task
#SBATCH --cpus-per-task=16               # CPU cores for self-play workloads
#SBATCH --mem=64GB                       # Total memory for the job
#SBATCH --time=12:00:00                  # Walltime (hh:mm:ss)
#SBATCH -o magnus-rl-%j.out              # Standard output
#SBATCH --mail-type=BEGIN,END,FAIL       # Email notifications

set -euo pipefail

module load python/3.10

# Use the directory where you ran `sbatch` as repo root
REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"

# Put the venv somewhere you actually own (home or project space)
VENV_DIR="${VENV_DIR:-$HOME/magnus_rl_env}"

echo "REPO_ROOT = $REPO_ROOT"
echo "VENV_DIR  = $VENV_DIR"

# Create / reuse venv
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment at $VENV_DIR ..."
    python3.10 -m venv "$VENV_DIR"
    source "$VENV_DIR/bin/activate"
    echo "Installing dependencies..."
    pip install --upgrade pip
    pip install -r "$REPO_ROOT/requirements.txt"
else
    echo "Using existing virtual environment at $VENV_DIR ..."
    source "$VENV_DIR/bin/activate"
fi

cd "$REPO_ROOT"

python -u 3600-agents/Magnus/train_weights.py \
    --opponent Magnus Bobby_2 Bobby_5_3 Tal\
    --iterations "${ITERATIONS:-150}" \
    --population "${POPULATION:-32}" \
    --games-per-candidate "${GAMES_PER_CANDIDATE:-40}" \
    --eval-games "${EVAL_GAMES:-160}" \
    --output "${OUTPUT_PATH:-3600-agents/Magnus/weights.json}"
