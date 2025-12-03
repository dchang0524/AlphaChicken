#!/bin/bash
# Aggregate results from batch game runs
# Run this after all SLURM jobs complete

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/batch_results"
MATCHES_DIR="${SCRIPT_DIR}/3600-agents/matches"

# Check if results directory exists
if [ ! -d "$RESULTS_DIR" ]; then
    echo "Error: batch_results directory not found!"
    echo "Make sure the games have completed."
    exit 1
fi

echo "Aggregating results from batch games..."
echo ""

python3 << EOF
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

results_dir = "${RESULTS_DIR}"
matches_data_dir = os.path.join(results_dir, "match_data")
depth_log_dir = os.path.join(results_dir, "depth_logs")

AGENT_NAME = "CPPHikaru_3"
OPPONENT = os.environ.get("OPPONENT", "Hikaru_3")

# Track statistics
wins = 0
losses = 0
ties = 0
errors = 0

# Separate real wins from opponent crash wins
real_wins = 0  # EGGS_LAID, BLOCKING_END, etc.
crash_wins = 0  # FAILED_INIT, CODE_CRASH, etc.
real_losses = 0  # Actual gameplay losses
crash_losses = 0  # We crashed/errored

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
if os.path.exists(depth_log_dir):
    for depth_file in Path(depth_log_dir).glob("game_*_depth.log"):
        game_id = depth_file.stem.replace("_depth", "")
        with open(depth_file) as f:
            for line in f:
                if line.startswith("DEPTH_LOG"):
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        turn_depth = parts[1].split(":")
                        if len(turn_depth) == 2:
                            try:
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
                            except ValueError:
                                pass

# Process match JSON files
if os.path.exists(matches_data_dir):
    match_files = sorted(Path(matches_data_dir).glob("game_*_match.json"))
    processed = 0

    for match_file in match_files:
        try:
            with open(match_file) as f:
                data = json.load(f)
            
            result = data.get("result")
            reason = data.get("reason", "UNKNOWN")
            
            # Determine if AGENT_NAME was player A or B from filename
            filename = match_file.name
            agent_is_a = True  # Since we're running as player A
            
            # Get final egg counts
            a_eggs = data.get("a_eggs_laid", [0])[-1] if data.get("a_eggs_laid") else 0
            b_eggs = data.get("b_eggs_laid", [0])[-1] if data.get("b_eggs_laid") else 0
            
            # Determine winner (PLAYER_A = 0, PLAYER_B = 1, TIE = 2, ERROR = 3)
            # Classify win reasons
            crash_reasons = ["FAILED_INIT", "CODE_CRASH", "MEMORY_ERROR", "TIMEOUT"]
            is_crash = reason in crash_reasons
            
            if result == 0:  # PLAYER_A wins
                wins += 1
                if is_crash:
                    crash_wins += 1
                else:
                    real_wins += 1
                egg_differences.append(a_eggs - b_eggs)
                win_reasons[reason] += 1
            elif result == 1:  # PLAYER_B wins
                losses += 1
                if is_crash:
                    crash_losses += 1
                else:
                    real_losses += 1
                egg_differences.append(b_eggs - a_eggs)
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
    print(f"\nWins: {wins} ({wins/processed*100:.1f}%)" if processed > 0 else "\nWins: 0")
    print(f"  Real Gameplay Wins: {real_wins}")
    print(f"  Opponent Crash Wins: {crash_wins}")
    print(f"\nLosses: {losses} ({losses/processed*100:.1f}%)" if processed > 0 else "\nLosses: 0")
    print(f"  Real Gameplay Losses: {real_losses}")
    print(f"  Agent Crash Losses: {crash_losses}")
    print(f"\nTies: {ties} ({ties/processed*100:.1f}%)" if processed > 0 else "\nTies: 0")
    print(f"Errors: {errors} ({errors/processed*100:.1f}%)" if processed > 0 else "\nErrors: 0")

    if wins + losses > 0:
        win_rate = wins / (wins + losses) * 100
        print(f"\nWin Rate (excluding ties): {win_rate:.1f}%")
    
    if real_wins + real_losses > 0:
        real_win_rate = real_wins / (real_wins + real_losses) * 100
        print(f"\n*** REAL GAMEPLAY WIN RATE: {real_win_rate:.1f}% ({real_wins} wins / {real_wins + real_losses} games) ***")
        print(f"    (Excludes opponent crashes)")

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
print(f"Results directory: {results_dir}")
print("="*70)

EOF

