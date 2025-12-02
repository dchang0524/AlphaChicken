# Potential Bugs in CPPHikaru_3

## 1. Trapdoor Undo Logic (CRITICAL)
**Location**: `game_rules.cpp:206-238`

**Issue**: After reversing perspective in undo, we subtract from `enemy_eggs_laid`, but we need to verify this is correct.

**Tracing**:
- Apply move: `enemy_eggs_laid += 4` (give opponent +4)
- Reverse perspective: swap(player_eggs_laid, enemy_eggs_laid)
  - Now player_eggs_laid = former enemy_eggs_laid (includes +4)
  - Now enemy_eggs_laid = former player_eggs_laid
- Undo: Reverse perspective first (swap back)
  - player_eggs_laid = what was enemy_eggs_laid (former player's eggs, no +4)
  - enemy_eggs_laid = what was player_eggs_laid (includes +4!)
- Restore: player_eggs_laid = saved value (correct)
- Subtract: enemy_eggs_laid -= 4 (correct!)

**Status**: Appears correct, but Python code might have different behavior. Need to verify.

## 2. Empty Moves Return Value
**Location**: `search.cpp:353-355`

**Issue**: When moves are empty, we return `Move()` (invalid move). Python returns `(-INF, None)`.

**Fix**: Return an invalid move is fine if handled correctly, but we should verify the caller handles this.

## 3. Terminal Evaluation After Perspective Switch
**Location**: `search.cpp:172-179`

**Issue**: After applying a move, perspective switches. We evaluate using `player_eggs_laid` and `enemy_eggs_laid` from current perspective, which should be correct. But we need to verify.

**Status**: Should be correct due to perspective swap.

## 4. Turn Counting
**Location**: `game_rules.cpp:189`

**Issue**: We decrement `turns_left_player` before perspective swap. After swap, this becomes `turns_left_enemy` from new perspective. Need to verify this matches Python.

## 5. State Conversion - Missing Fields?
**Location**: `agent.py:105-176`, `cpphikaru3.cpp:62-79`

**Issue**: Need to verify all Board fields are correctly converted to GameState.

**Checked fields**:
- ✅ Positions (player, enemy, spawns)
- ✅ Eggs laid
- ✅ Turds left
- ✅ Even chicken parity
- ✅ Turn count
- ✅ Turns left
- ✅ is_as_turn
- ✅ Time
- ✅ Eggs/turds bitboards
- ✅ Known traps

**Status**: Looks complete.

## 6. Zobrist Hashing
**Location**: `zobrist.cpp`

**Issue**: Need to verify hash includes all necessary state for transposition table.

**Checked**:
- ✅ Player/enemy chicken positions
- ✅ Eggs/turds (both players)
- ✅ Known traps
- ✅ Side to move

**Status**: Looks complete.

## 7. Move Validation - Turd Distance Check
**Location**: `game_rules.cpp:111-115`

**Issue**: Need to verify `can_lay_turd_at_loc` matches Python logic exactly.

## 8. Evaluation Function - Perspective
**Location**: `evaluation.cpp`, `search.cpp:184`

**Issue**: Evaluation is called after perspective switches. Need to verify evaluation uses correct perspective.

**Status**: Should be correct - evaluation uses `state.player_eggs_laid` which is correct from current perspective.

## 9. Risk Calculation - Potential Lists Check
**Location**: `search.cpp:453-481`

**Issue**: We check if square is in potential_even/potential_odd lists. This matches Python, but we disabled visited squares tracking. This might cause issues.

**Status**: Currently disabled to match Python exactly.

## 10. Corner Reward
**Location**: `game_rules.cpp:164`

**Issue**: CORNER_REWARD = 3. Need to verify this matches Python's game_map.CORNER_REWARD.

**Status**: ✅ Matches (3).

