# Batch Game Testing Script

## Overview
The `run_batch_games.sh` script runs multiple games in parallel and provides detailed statistics including search depth analysis.

## Usage

```bash
./run_batch_games.sh [opponent_name] [num_games]
```

### Examples

```bash
# Run 100 games against Hikaru_3 (default)
./run_batch_games.sh Hikaru_3 100

# Run 50 games against a different opponent
./run_batch_games.sh Magnus 50

# Run 200 games
./run_batch_games.sh Hikaru_3 200
```

## Output

The script creates a timestamped results directory with:
- **logs/**: Individual game log files
- **depth_logs/**: Depth tracking per game
- **Summary printed to console** with:
  - Win/Loss/Tie statistics
  - Average egg differences
  - Win reasons
  - **Average search depth by move range** (1-10, 11-20, 21-30, 31-40)

## Requirements

- Python 3
- GNU parallel (recommended) or xargs for parallel execution
- The agent must be compiled and ready

## Parallel Execution

The script will automatically detect available parallel tools:
1. **GNU parallel** (preferred) - uses all CPU cores
2. **xargs** - fallback parallel execution
3. **Sequential** - if neither is available

## Example Output

```
BATCH GAME RESULTS SUMMARY
======================================================================
Total Games: 100
Agent: CPPHikaru_3
Opponent: Hikaru_3

Wins: 65 (65.0%)
Losses: 30 (30.0%)
Ties: 5 (5.0%)
Errors: 0 (0.0%)

Win Rate (excluding ties): 68.4%

Average Egg Difference: 2.3
  (Positive = CPPHikaru_3 advantage, Negative = Hikaru_3 advantage)

Win Reasons (when CPPHikaru_3 won):
  EGGS_LAID: 45
  BLOCKING_END: 20

======================================================================
AVERAGE SEARCH DEPTH BY MOVE RANGE
======================================================================

Moves 1-10:
  Average Depth: 12.5
  Min Depth: 8
  Max Depth: 18
  Total Samples: 1000

Moves 11-20:
  Average Depth: 15.2
  Min Depth: 10
  Max Depth: 22
  Total Samples: 1000

...
```

