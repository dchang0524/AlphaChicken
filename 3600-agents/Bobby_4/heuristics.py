import math
from typing import List, Tuple

from game import board as board_mod  # type: ignore
from .voronoi import VoronoiInfo, OWNER_ME, OWNER_OPP  # adjust import path as needed
from .hiddenMarkov import TrapdoorBelief
from game.enums import Direction, MoveType  # type: ignore

INF = 10 ** 8

def evaluate(cur_board: board_mod.Board, vor: VoronoiInfo, trap_belief: TrapdoorBelief) -> float:
    # 1. Terminal Check
    if cur_board.is_game_over():
        my_eggs  = cur_board.chicken_player.eggs_laid
        opp_eggs = cur_board.chicken_enemy.eggs_laid
        if my_eggs > opp_eggs: return INF
        elif my_eggs < opp_eggs: return -INF
        else: return 0.0

    # 2. Setup Data
    my_eggs  = cur_board.chicken_player.eggs_laid
    opp_eggs = cur_board.chicken_enemy.eggs_laid
    mat_diff = my_eggs - opp_eggs
    
    # 3. Calculate Phase (1.0 = Start, 0.0 = End)
    moves_left = cur_board.MAX_TURNS - cur_board.turn_count
    phase = moves_left / cur_board.MAX_TURNS 

    # 4. Base Weights (Use your current or tuned values here)
    # These are just placeholders; keep your existing logic for these if you have it!
    W_MAT   = 25.0 
    W_SPACE = 5.0  
    W_FRAG  = 3.0
    W_FRONTIER = 1.0

    # --- [THE FIX] ENDGAME PANIC LOGIC ---
    # Changed from 15 to 8.
    # 8 turns is enough time to traverse the board (8x8) to get an egg,
    # but ensures we don't panic during the mid-late game.
    PANIC_THRESHOLD = 8 
    
    if moves_left <= PANIC_THRESHOLD:
        # A) DECAY SPACE VALUE
        decay_factor = max(0.0, moves_left / float(PANIC_THRESHOLD))
        
        W_SPACE    *= decay_factor
        W_FRAG     *= decay_factor
        W_FRONTIER *= 0.0 

        # B) PANIC BOOST
        # Only panic if we are actually losing or tied.
        if mat_diff <= 0:
            W_MAT *= 100.0
    # --------------------------------------

    # 5. Openness Logic (Your existing logic)
    max_contested = 8.0
    openness = 0.0
    if max_contested > 0:
        openness = max(0.0, min(1.0, vor.contested / max_contested))
    
    # Interpolate Space/Mat based on openness/phase (Your existing logic)
    # Note: We apply the panic modifiers to the base W_SPACE/W_MAT first
    
    w_space_final = W_SPACE * (1.0 + openness) # Example interpolation
    w_mat_final   = W_MAT + (1.0 - phase) * 10.0 

    # 6. Calculate Final Score
    # Note: We use the Modified Weights here
    score = (w_mat_final * mat_diff) + \
            (w_space_final * vor.vor_score) - \
            (W_FRAG * openness * vor.frag_score) - \
            (W_FRONTIER * vor.max_contested_dist)

    return score

Move = Tuple[Direction, MoveType]


def move_order(
    cur_board: board_mod.Board,
    moves: List[Move],
    vor: VoronoiInfo,
) -> List[Move]:
    """
    Order moves using:
      - Direction priority based on # of contested squares in that region (if > 0)
      - If total contested == 0: drop TURD moves, then random order
      - If there is any EGG move: drop PLAIN moves
      - Generally: EGG > PLAIN > TURD
      - If we are closer to center: TURD > PLAIN
      - If near frontier (min_contested_dist <= 2): TURD can even beat EGG
    """

    if not moves:
        return moves

    dim = cur_board.game_map.MAP_SIZE

    # ----------------------------
    # 1) Basic filtering
    # ----------------------------

    filtered = list(moves)

    total_contested = getattr(vor, "contested", 0)

    # If no contested squares: drop TURD moves
    if total_contested == 0:
        filtered = [mv for mv in filtered if mv[1] != MoveType.TURD]

    # If there is any EGG move, drop PLAIN moves
    has_egg = any(mt == MoveType.EGG for _, mt in filtered)
    if has_egg:
        filtered = [mv for mv in filtered if mv[1] != MoveType.PLAIN]

    # If everything got filtered away, fall back to original moves
    if not filtered:
        filtered = list(moves)

    # ----------------------------
    # 2) Directional contested counts
    # ----------------------------

    contested_by_dir: dict[Direction, int] = {
        Direction.LEFT:  vor.contested_left,
        Direction.RIGHT: vor.contested_right,
        Direction.UP:    vor.contested_up,
        Direction.DOWN:  vor.contested_down,
    }

    # ----------------------------
    # 3) Context flags: near frontier, closer to center
    # ----------------------------

    # Near frontier if closest contested square is very close
    min_frontier_dist = vor.min_contested_dist
    near_frontier = (
        min_frontier_dist is not None
        and min_frontier_dist >= 0
        and min_frontier_dist <= 2
    )

    def manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    center = (dim // 2, dim // 2)

    my_pos  = cur_board.chicken_player.get_location()

    my_center_dist  = manhattan(my_pos, center)

    # ----------------------------
    # 4) Scoring each move
    # ----------------------------

    # Base priority: EGG > PLAIN > TURD
    BASE_EGG   = 3.0
    BASE_PLAIN = 2.0
    BASE_TURD  = 1.0

    # How much contested counts by direction matter
    DIR_WEIGHT = 0.5

    # TURD buffs
    TURD_CENTER_BONUS   = 2.0   # TURD > PLAIN if we’re closer to center
    TURD_FRONTIER_BONUS = 3.0   # TURD can rival/beat EGG near frontier

    scored: List[tuple[float, Move]] = []

    for mv in filtered:
        direction, mtype = mv

        # 4a) Base by move type
        if mtype == MoveType.EGG:
            score = BASE_EGG
        elif mtype == MoveType.PLAIN:
            score = BASE_PLAIN
        else:  # MoveType.TURD
            score = BASE_TURD

        # 4b) Direction: more contested squares in that direction → more priority
        dir_contested = contested_by_dir.get(direction, 0)
        score += DIR_WEIGHT * dir_contested

        # 4c) Contextual TURD buffs
        if mtype == MoveType.TURD:
            if my_center_dist <= 2:
                # TURD > PLAIN when we control center
                score += TURD_CENTER_BONUS
            if near_frontier:
                # TURD can beat EGG near frontier
                score += TURD_FRONTIER_BONUS

        # Tiny jitter to break ties
        scored.append((score, mv))

    # ----------------------------
    # 5) Sort best → worst
    # ----------------------------

    scored.sort(key=lambda x: x[0], reverse=True)
    return [mv for _, mv in scored]

