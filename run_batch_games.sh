#!/bin/bash

# Batch game runner for testing CPPHikaru_3
# Usage: ./run_batch_games.sh [opponent_name] [num_games]
# Example: ./run_batch_games.sh Hikaru_3 100

set -e

# Configuration
OPPONENT="${1:-Hikaru_3}"
NUM_GAMES="${2:-100}"
AGENT_NAME="CPPHikaru_3"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/batch_results_$(date +%Y%m%d_%H%M%S)"
GAME_DIR="${SCRIPT_DIR}/engine"

mkdir -p "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}/logs"
mkdir -p "${OUTPUT_DIR}/depth_logs"

echo "Running ${NUM_GAMES} games: ${AGENT_NAME} vs ${OPPONENT}"
echo "Results will be saved to: ${OUTPUT_DIR}"
echo ""

# Function to run a single game
run_game() {
    local game_id=$1
    local log_file="${OUTPUT_DIR}/logs/game_${game_id}.log"
    local depth_file="${OUTPUT_DIR}/depth_logs/game_${game_id}_depth.log"
    local stderr_file="${OUTPUT_DIR}/logs/game_${game_id}_stderr.log"
    
    # Run game and capture stderr (which contains depth logs)
    cd "${GAME_DIR}"
    python3 run_local_agents.py "${AGENT_NAME}" "${OPPONENT}" \
        > "${log_file}" 2> "${stderr_file}" || true
    
    # Extract depth logs from stderr
    grep "DEPTH_LOG" "${stderr_file}" > "${depth_file}" || touch "${depth_file}"
    
    echo "Game ${game_id} completed"
}

# Export function for parallel execution
export -f run_game
export OUTPUT_DIR AGENT_NAME OPPONENT GAME_DIR

# Run games in parallel (using GNU parallel if available, otherwise xargs)
if command -v parallel &> /dev/null; then
    echo "Using GNU parallel for execution..."
    seq 1 "${NUM_GAMES}" | parallel -j "$(nproc)" run_game {}
elif command -v xargs &> /dev/null; then
    echo "Using xargs for parallel execution..."
    seq 1 "${NUM_GAMES}" | xargs -n 1 -P "$(nproc)" -I {} bash -c 'run_game {}'
else
    echo "Running games sequentially (no parallel tool found)..."
    for i in $(seq 1 "${NUM_GAMES}"); do
        run_game "${i}"
    done
fi

echo ""
echo "All games completed. Analyzing results..."

# Python script to analyze results
python3 << EOF
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

output_dir = "${OUTPUT_DIR}"
matches_dir = "${SCRIPT_DIR}/3600-agents/matches"

# Track statistics
wins = 0
losses = 0
ties = 0
errors = 0

win_reasons = defaultdict(int)
egg_differences = []

# Track depth by move range
depth_by_range = {
    "1-10": [],
    "11-20": [],
    "21-30": [],
    "31-40": []
}

# Process each game's depth log
depth_log_dir = os.path.join(output_dir, "depth_logs")
for depth_file in Path(depth_log_dir).glob("game_*_depth.log"):
    game_id = depth_file.stem.replace("_depth", "")
    with open(depth_file) as f:
        for line in f:
            if line.startswith("DEPTH_LOG"):
                parts = line.strip().split()
                if len(parts) >= 2:
                    turn_depth = parts[1].split(":")
                    if len(turn_depth) == 2:
                        turn = int(turn_depth[0])
                        depth = int(turn_depth[1])
                        
                        if turn <= 10:
                            depth_by_range["1-10"].append(depth)
                        elif turn <= 20:
                            depth_by_range["11-20"].append(depth)
                        elif turn <= 30:
                            depth_by_range["21-30"].append(depth)
                        else:
                            depth_by_range["31-40"].append(depth)

# Process match JSON files - get all matches for this pairing
match_files = sorted(Path(matches_dir).glob(f"{AGENT_NAME}_{OPPONENT}_*.json"))
# Also try reverse order
match_files.extend(sorted(Path(matches_dir).glob(f"{OPPONENT}_{AGENT_NAME}_*.json")))
processed = 0

# Track which files we've seen to avoid double counting
seen_files = set()

for match_file in match_files:
    if match_file in seen_files:
        continue
    seen_files.add(match_file)
    
    try:
        with open(match_file) as f:
            data = json.load(f)
        
        result = data.get("result")
        reason = data.get("reason", "UNKNOWN")
        
        # Determine if AGENT_NAME was player A or B from filename
        filename = match_file.name
        agent_is_a = filename.startswith(f"{AGENT_NAME}_")
        
        # Get final egg counts
        a_eggs = data.get("a_eggs_laid", [0])[-1] if data.get("a_eggs_laid") else 0
        b_eggs = data.get("b_eggs_laid", [0])[-1] if data.get("b_eggs_laid") else 0
        
        # Determine winner (PLAYER_A = 0, PLAYER_B = 1, TIE = 2, ERROR = 3)
        if result == 0:  # PLAYER_A wins
            if agent_is_a:
                wins += 1
                egg_differences.append(a_eggs - b_eggs)
                win_reasons[reason] += 1
            else:
                losses += 1
                egg_differences.append(a_eggs - b_eggs)  # Opponent won
        elif result == 1:  # PLAYER_B wins
            if agent_is_a:
                losses += 1
                egg_differences.append(b_eggs - a_eggs)  # Opponent won
            else:
                wins += 1
                egg_differences.append(b_eggs - a_eggs)
                win_reasons[reason] += 1
        elif result == 2:  # TIE
            ties += 1
            egg_differences.append(0)
        else:  # ERROR
            errors += 1
        
        processed += 1
        
    except Exception as e:
        print(f"Error processing {match_file}: {e}", file=sys.stderr)
        continue

# Print summary
print("\n" + "="*70)
print("BATCH GAME RESULTS SUMMARY")
print("="*70)
print(f"\nTotal Games: {processed}")
print(f"Agent: {AGENT_NAME}")
print(f"Opponent: {OPPONENT}")
print(f"\nWins: {wins} ({wins/processed*100:.1f}%)")
print(f"Losses: {losses} ({losses/processed*100:.1f}%)")
print(f"Ties: {ties} ({ties/processed*100:.1f}%)")
print(f"Errors: {errors} ({errors/processed*100:.1f}%)")

if wins + losses > 0:
    win_rate = wins / (wins + losses) * 100
    print(f"\nWin Rate (excluding ties): {win_rate:.1f}%")

if egg_differences:
    avg_diff = sum(egg_differences) / len(egg_differences)
    print(f"\nAverage Egg Difference: {avg_diff:.2f}")
    print(f"  (Positive = {AGENT_NAME} advantage, Negative = {OPPONENT} advantage)")

if win_reasons:
    print(f"\nWin Reasons (when {AGENT_NAME} won):")
    for reason, count in sorted(win_reasons.items(), key=lambda x: -x[1]):
        print(f"  {reason}: {count}")

# Average depth by move range
print(f"\n" + "="*70)
print("AVERAGE SEARCH DEPTH BY MOVE RANGE")
print("="*70)

for move_range, depths in depth_by_range.items():
    if depths:
        avg_depth = sum(depths) / len(depths)
        max_depth = max(depths)
        min_depth = min(depths)
        print(f"\nMoves {move_range}:")
        print(f"  Average Depth: {avg_depth:.2f}")
        print(f"  Min Depth: {min_depth}")
        print(f"  Max Depth: {max_depth}")
        print(f"  Total Samples: {len(depths)}")
    else:
        print(f"\nMoves {move_range}: No depth data available")

print("\n" + "="*70)
print(f"Full results saved to: {output_dir}")
print("="*70)

EOF

echo ""
echo "Analysis complete! Results saved to ${OUTPUT_DIR}"

