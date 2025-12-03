# Log Analysis Summary

## Critical Bugs Found

### 1. ✅ FIXED: Known Trapdoors Not Blocked
**Issue**: `is_cell_blocked()` didn't check for known trapdoors, allowing moves onto them.
**Fix**: Added `known_traps` parameter to `is_cell_blocked()`, `is_valid_move()`, and `get_valid_moves()`, and block cells in `known_traps`.
**Impact**: Agent was stepping on known trapdoors repeatedly (e.g., turn 4 and 10).

### 2. ⚠️ POTENTIAL: Search Evaluation Too Optimistic When Losing

**Evidence from logs:**
- Turn 6 (down 4 eggs): `base_eval: -45.47`, but `search_eval: -1.55` (way too optimistic)
- Turn 10 (down 5 eggs): `base_eval: -20.66`, `AFTER_MOVE: -37.5` (worse!), but `search_eval: -31.25`

**Possible causes:**
1. Transposition table storing values from wrong perspective
2. Scenario weighting giving too much weight to unlikely good outcomes
3. Search finding unrealistic recovery paths at high depth
4. Terminal evaluation perspective issue (though code looks correct)

**Next steps to investigate:**
- Add logging to see what scenarios are being considered and their weights
- Check if TT entries are being used correctly with perspective switches
- Verify terminal evaluation is correct after perspective switches

### 3. ⚠️ POTENTIAL: Moves That Worsen Position Are Chosen

**Evidence:**
- Turn 10: `BEFORE_MOVE: -20.66`, `AFTER_MOVE: -37.5` (move makes position worse by 16.84 points)
- Yet `search_eval: -31.25` suggests the move is acceptable

**Possible causes:**
1. Search sees good positions deeper in tree that aren't actually achievable
2. Evaluation function incorrectly values space over material when losing
3. Sign error in how search values are propagated

### 4. ⚠️ POTENTIAL: Risk Calculation Disabled at Root

**Code location**: `search.cpp:481`
```cpp
float delta_risk_at_root = 0.0f;  // TEMPORARILY DISABLE
```

**Impact**: Trapdoor risk is not penalized at root level moves, only in recursive calls. This matches Python's behavior (which also doesn't apply risk at root), but might be contributing to poor move selection.

## Comparison with Python Agent

### Terminal Evaluation
- ✅ Matches: Both return `INF`/`-INF`/`0` based on egg count
- ✅ Perspective handling: Both use current player's perspective

### Negamax Structure
- ✅ Matches: Both negate recursive call: `child_val = -negamax(...)`
- ✅ Matches: Both use `child_cum_risk = -(cum_risk + delta_risk)`

### Risk Calculation
- ⚠️ Discrepancy: Python doesn't apply risk at root either, but C++ has it explicitly disabled
- ✅ Matches: Both exclude risk for squares in `potential_even`/`potential_odd` lists

## Recommendations

1. **Test with trapdoor fix**: The known trapdoor bug was critical - test if this alone fixes the win rate
2. **Add scenario logging**: Log which scenarios are considered and their weights to understand why search_eval differs from base_eval
3. **Verify TT perspective**: Ensure transposition table entries are stored/retrieved with correct perspective
4. **Compare with Python**: Run same positions through Python agent and compare search values

## Next Debugging Steps

1. Add logging to show:
   - Scenario weights and counts
   - TT hit rates and values
   - Terminal node evaluations
   
2. Test specific positions:
   - Position where agent is down 4 eggs (turn 6)
   - Compare C++ vs Python evaluation for same position
   
3. Check if issue is evaluation or search:
   - If `base_eval` matches Python but `search_eval` doesn't → search bug
   - If `base_eval` doesn't match Python → evaluation bug

