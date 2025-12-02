#!/bin/bash
# Script to run a single game - used by SLURM job array
# Usage: run_single_game.sh <game_id> <opponent>

set -e

GAME_ID=$1
OPPONENT="${2:-Hikaru_3}"
AGENT_NAME="CPPHikaru_3"

# Get directory where script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# Set library path
export LD_LIBRARY_PATH="${SCRIPT_DIR}/3600-agents/CPPHikaru_3:${LD_LIBRARY_PATH}"

# Create output directories
mkdir -p batch_results/game_logs
mkdir -p batch_results/depth_logs
mkdir -p batch_results/match_data

# Run single game
LOG_FILE="batch_results/game_logs/game_${GAME_ID}.log"
DEPTH_FILE="batch_results/depth_logs/game_${GAME_ID}_depth.log"
STDERR_FILE="batch_results/game_logs/game_${GAME_ID}_stderr.log"

cd engine
python3 run_local_agents.py "${AGENT_NAME}" "${OPPONENT}" \
    > "${SCRIPT_DIR}/${LOG_FILE}" 2> "${SCRIPT_DIR}/${STDERR_FILE}" || true

# Extract depth logs from stderr
grep "DEPTH_LOG" "${SCRIPT_DIR}/${STDERR_FILE}" > "${SCRIPT_DIR}/${DEPTH_FILE}" || touch "${SCRIPT_DIR}/${DEPTH_FILE}"

# Return to script directory
cd "${SCRIPT_DIR}"

# Find the match JSON file created
# The file is created in 3600-agents/matches/ relative to top level (parent of engine)
MATCHES_DIR="${SCRIPT_DIR}/3600-agents/matches"
mkdir -p "${MATCHES_DIR}"

# Wait a moment for file to be written (filesystem sync)
sleep 1

# Find most recently created match file for this pairing (by modification time, most portable method)
# Try both orders: AGENT_OPPONENT and OPPONENT_AGENT
MATCH_FILE=$(ls -t "${MATCHES_DIR}/${AGENT_NAME}_${OPPONENT}_"*.json 2>/dev/null | head -1 || true)

if [ -z "$MATCH_FILE" ] || [ ! -f "$MATCH_FILE" ]; then
    MATCH_FILE=$(ls -t "${MATCHES_DIR}/${OPPONENT}_${AGENT_NAME}_"*.json 2>/dev/null | head -1 || true)
fi

if [ -n "$MATCH_FILE" ] && [ -f "$MATCH_FILE" ]; then
    # Copy match file with game ID
    cp "$MATCH_FILE" "${SCRIPT_DIR}/batch_results/match_data/game_${GAME_ID}_match.json"
    echo "  ✓ Match file saved: game_${GAME_ID}_match.json"
else
    echo "  ⚠ Warning: No match file found in ${MATCHES_DIR}"
    echo "  Available files: $(ls -1 ${MATCHES_DIR}/*.json 2>/dev/null | wc -l) JSON files"
    echo "  Check game log: tail ${LOG_FILE}"
    echo "  Check for errors: tail ${STDERR_FILE}"
fi

echo "Game ${GAME_ID} completed"

