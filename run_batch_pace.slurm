#!/bin/bash
#SBATCH --job-name=CPPHikaru_Batch      # Job name
#SBATCH -N1 --ntasks-per-node=1         # One node, single task
#SBATCH --cpus-per-task=16              # CPU cores for parallel games
#SBATCH --mem=32GB                      # Total memory for the job
#SBATCH --time=12:00:00                 # Walltime (hh:mm:ss)
#SBATCH -o cpphikaru-batch-%j.out       # Standard output
#SBATCH -e cpphikaru-batch-%j.err       # Standard error
#SBATCH --mail-type=BEGIN,END,FAIL      # Email notifications

set -euo pipefail

# Load Python 3.10+ (required for match/case syntax)
module load python/3.10 || module load python/3.11 || module load python/3.12
python3 --version

# Use the directory where you ran `sbatch` as repo root
REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"

# Optional: Create/use virtual environment
VENV_DIR="${VENV_DIR:-$HOME/cpphikaru_env}"

echo "REPO_ROOT = $REPO_ROOT"
echo "VENV_DIR  = $VENV_DIR"

# Create / reuse venv if needed
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment at $VENV_DIR ..."
    python3.10 -m venv "$VENV_DIR"
    source "$VENV_DIR/bin/activate"
    echo "Installing dependencies..."
    pip install --upgrade pip
    if [ -f "$REPO_ROOT/requirements.txt" ]; then
        pip install -r "$REPO_ROOT/requirements.txt"
    else
        pip install numpy psutil
    fi
else
    echo "Using existing virtual environment at $VENV_DIR ..."
    source "$VENV_DIR/bin/activate"
fi

cd "$REPO_ROOT"

# Set library path for Linux .so
export LD_LIBRARY_PATH="${REPO_ROOT}/3600-agents/CPPHikaru_3:${LD_LIBRARY_PATH}"

# Setup if library doesn't exist
if [ ! -f "3600-agents/CPPHikaru_3/libcpphikaru3.so" ]; then
    echo "Compiling C++ library..."
    cd 3600-agents/CPPHikaru_3
    if [ -f "Makefile" ]; then
        make clean
        make
    else
        echo "ERROR: No Makefile found!"
        exit 1
    fi
    cd ../..
fi

# Run batch games
# Usage: sbatch run_batch_pace.slurm [opponent] [num_games]
OPPONENT="${1:-Hikaru_3}"
NUM_GAMES="${2:-100}"

echo ""
echo "Running ${NUM_GAMES} games: CPPHikaru_3 vs ${OPPONENT}"
echo ""

bash run_batch_games.sh "${OPPONENT}" "${NUM_GAMES}"

echo ""
echo "Job completed! Check batch_results_* directory for results."

