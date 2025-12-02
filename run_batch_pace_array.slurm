#!/bin/bash
#SBATCH --job-name=CPPHikaru_Batch    # Job name
#SBATCH --array=1-100                  # Array of 100 jobs (game IDs 1-100)
#SBATCH --ntasks-per-node=1            # One task per job
#SBATCH --cpus-per-task=1              # Each game uses 1 CPU
#SBATCH --mem=4GB                      # Memory per game
#SBATCH --time=0:20:00                 # 20 minutes per game (should be enough)
#SBATCH -o cpphikaru-%A_%a.out         # %A = job ID, %a = array index
#SBATCH -e cpphikaru-%A_%a.err         # Error log per job
#SBATCH --mail-type=END,FAIL           # Email when job finishes or fails

set -euo pipefail

# Load Python 3.10+ (required for match/case syntax)
module load python/3.10 || module load python/3.11 || module load python/3.12
python3 --version

# Use the directory where you ran `sbatch` as repo root
REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"

# Optional: Create/use virtual environment
VENV_DIR="${VENV_DIR:-$HOME/cpphikaru_env}"

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
    source "$VENV_DIR/bin/activate"
fi

cd "$REPO_ROOT"

# Set library path
export LD_LIBRARY_PATH="${REPO_ROOT}/3600-agents/CPPHikaru_3:${LD_LIBRARY_PATH}"

# Get array task ID (this is the game number)
GAME_ID=${SLURM_ARRAY_TASK_ID}

# Get opponent from environment or use default
OPPONENT="${OPPONENT:-Hikaru_3}"

# Get player (A or B) from environment or use default
PLAYER="${PLAYER:-A}"

echo "Running game ${GAME_ID}: CPPHikaru_3 vs ${OPPONENT} (as player ${PLAYER})"

# Run single game
bash run_single_game.sh "${GAME_ID}" "${OPPONENT}" "${PLAYER}"

echo "Game ${GAME_ID} completed successfully"

