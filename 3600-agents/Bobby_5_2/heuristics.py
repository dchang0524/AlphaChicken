import math
from typing import List, Tuple

from game import board as board_mod  # type: ignore
from .voronoi import VoronoiInfo, OWNER_ME, OWNER_OPP  # adjust import path as needed
from .hiddenMarkov import TrapdoorBelief
from game.enums import Direction, MoveType  # type: ignore

INF = 10 ** 8

def evaluate(cur_board: board_mod.Board, vor: VoronoiInfo, trap_belief : TrapdoorBelief) -> float:
    """
    Bobby's scalar evaluation.
    Always from *my* POV (current chicken_player), regardless of depth.

    Components:
      - mat_term:       egg difference, with weight increasing as turns run out
      - space_term:     Voronoi score (my - opp) with weight tied to openness
      - risk_term:      regional trap risk (my_mass - opp_mass) over Voronoi
      - frag_term:      penalty for fragmented frontiers (shape-based)
      - frontier_term:  penalty for being far from your frontier (max distance)
      - frontier_bonus: small bonus for having some frontier when the game is open
    """
    if cur_board.is_game_over():
        my_eggs  = cur_board.chicken_player.eggs_laid
        opp_eggs = cur_board.chicken_enemy.eggs_laid
        if my_eggs > opp_eggs:
            return INF
        elif my_eggs < opp_eggs:
            return -INF
        else:
            return 0

    dim = cur_board.game_map.MAP_SIZE

    # --- Basic features: material and space ---

    my_eggs  = cur_board.chicken_player.eggs_laid
    opp_eggs = cur_board.chicken_enemy.eggs_laid
    mat_diff = my_eggs - opp_eggs

    # Voronoi score: my_voronoi - opp_voronoi (already POV-relative)
    space_score = vor.vor_score

    # --- Phase terms: time-based for material, structure-based for space/risk ---

    # Time-to-go: material matters more as we approach the end.
    moves_left  = cur_board.MAX_TURNS - cur_board.turn_count
    total_moves = cur_board.MAX_TURNS
    phase_mat   = max(0.0, min(1.0, moves_left / total_moves))  # 1 early, 0 late

    # Openness: how much frontier there is. 0 = closed, 1 = very open.
    # There can't be more than ~8 contested squares; normalize by that.
    max_contested = 8.0
    openness = 0.0
    if max_contested > 0:
        openness = max(0.0, min(1.0, vor.contested / max_contested))

    # --- Weights in egg units (tunable) ---

    # Space: important when open, but never completely zero.
    W_SPACE_MIN = 5   # closed
    W_SPACE_MAX = 30.0   # very open

    # Material: always matters, but ramps up hard toward the end.
    W_MAT_MIN   = 5.0   # early
    W_MAT_MAX   = 25.0  # late

    # # Trap risk: matters most when the board is open and mobility is high.
    # W_RISK_MIN  = 0.5
    # W_RISK_MAX  = 3.0

    # Fragmentation: how bad it is if contested squares are spatially split.
    # Assumes vor.frag_score ∈ [0,1]
    W_FRAG      = 3.0

    # Frontier distance: how bad it is if I'm far from my most distant frontier.
    # Assumes vor.max_contested_dist = max dist(from my chicken, to any contested square)
    W_FRONTIER_DIST = 1.5

    # Frontier closeness bonus.
    FRONTIER_COEFF = 0.5

    # Interpolate weights
    w_space = W_SPACE_MIN + (1 - openness) * (W_SPACE_MAX - W_SPACE_MIN)
    if openness == 0.00:
        w_space = 0.0
    w_mat   = W_MAT_MIN   + (1.0 - phase_mat) * (W_MAT_MAX - W_MAT_MIN)
    # w_risk  = W_RISK_MIN  + openness * (W_RISK_MAX - W_RISK_MIN)

    # --- Fragmentation & frontier geometry ---

    # 1) Fragmentation score: 0 (all frontier in one blob) → 1 (highly fragmented).
    # Penalize more when the board is open.
    frag_score = vor.frag_score  # fail-safe if not set
    frag_score = max(0.0, min(1.0, frag_score))
    frag_term  = -W_FRAG * openness * frag_score

    # 2) Max contested distance from *my chicken* to any contested square.
    # Encourage being close to the whole frontier: large distance = bad.
    max_contested_dist = vor.max_contested_dist

    frontier_dist_term = -FRONTIER_COEFF * (1 - frag_score) * max_contested_dist

    PANIC_THRESHOLD = 8 
    
    # if moves_left <= PANIC_THRESHOLD:
    #     # A) DECAY SPACE VALUE
    #     decay_factor = max(0.0, moves_left / float(PANIC_THRESHOLD))
        
    #     w_space    *= decay_factor
    #     W_FRAG     *= decay_factor
    #     FRONTIER_COEFF *= 0.0 

    #     # B) PANIC BOOST
    #     # Only panic if we are actually losing or tied.
    #     if mat_diff <= 0:
    #         w_mat *= 100.0

    CLOSEST_EGG_COEFF = (w_mat - 5) * 0.1 if moves_left <= PANIC_THRESHOLD or openness == 0 else 0.0
    egg_dist = vor.min_egg_dist if vor.min_egg_dist <= 63 else 0
    # --- Combine everything ---

    space_term = w_space * space_score
    mat_term   = w_mat   * mat_diff

    return space_term + mat_term + frag_term + frontier_dist_term - CLOSEST_EGG_COEFF * egg_dist * 0.25

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

