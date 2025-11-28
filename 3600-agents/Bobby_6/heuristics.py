import math
from typing import List, Tuple, Dict

from game import board as board_mod  # type: ignore
from .voronoi import VoronoiInfo, OWNER_ME, OWNER_OPP  # adjust import path as needed
from .hiddenMarkov import TrapdoorBelief
from game.enums import Direction, MoveType  # type: ignore
from .weights import HeuristicWeights

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
    moves_left  = cur_board.turns_left_player
    total_moves = cur_board.MAX_TURNS
    phase_mat   = max(0.0, min(1.0, moves_left / total_moves))  # 1 early, 0 late

    # Openness: how much frontier there is. 0 = closed, 1 = very open.
    # There can't be more than ~8 contested squares; normalize by that.
    max_contested = 8.0
    openness = 0.0
    if max_contested > 0:
        openness = max(0.0, min(1.0, vor.contested / max_contested))

    # --- Weights in egg units (tunable) ---
    # Using HeuristicWeights class


    # Interpolate weights
    w_space = HeuristicWeights.W_SPACE_MIN + phase_mat * (HeuristicWeights.W_SPACE_MAX - HeuristicWeights.W_SPACE_MIN)
    if openness == 0.00:
        w_space = 0.0
    w_mat   = HeuristicWeights.W_MAT_MIN   + (1.0 - phase_mat) * (HeuristicWeights.W_MAT_MAX - HeuristicWeights.W_MAT_MIN)

    # --- Fragmentation & frontier geometry ---

    # 1) Fragmentation score: 0 (all frontier in one blob) → 1 (highly fragmented).
    # Penalize more when the board is open.
    frag_score = vor.frag_score  # fail-safe if not set
    frag_score = max(0.0, min(1.0, frag_score))
    frag_term  = -HeuristicWeights.W_FRAG * openness * frag_score

    # 2) Max contested distance from *my chicken* to any contested square.
    # Encourage being close to the whole frontier: large distance = bad.
    frontier_dist_term = -HeuristicWeights.FRONTIER_COEFF * (1 - frag_score) * vor.average_contested_dist

    PANIC_THRESHOLD = HeuristicWeights.PANIC_THRESHOLD 
    
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
    W_BLOCKS = mat_term / 5


    score = space_term + mat_term + frag_term + frontier_dist_term - CLOSEST_EGG_COEFF * egg_dist * 0.25 + W_BLOCKS * (vor.my_owned - vor.opp_owned)
    
    # Weighted contested term
    # Reward holding contested squares that lead to large regions
    score += HeuristicWeights.W_WEIGHTED_CONTESTED * vor.weighted_contested

    if cur_board.chicken_player.eggs_laid + (moves_left-vor.min_egg_dist) / 2 < cur_board.chicken_enemy.eggs_laid:
        score -= INF
        score += (my_eggs - opp_eggs) * 100
    return score


def get_weight_gradient(cur_board: board_mod.Board, vor: VoronoiInfo, trap_belief : TrapdoorBelief) -> Dict[str, float]:
    """
    Returns the gradient of the evaluation function with respect to the weights.
    Assumes linear combination of features.
    """
    if cur_board.is_game_over():
        return {}

    dim = cur_board.game_map.MAP_SIZE

    # --- Basic features: material and space ---
    my_eggs  = cur_board.chicken_player.eggs_laid
    opp_eggs = cur_board.chicken_enemy.eggs_laid
    mat_diff = my_eggs - opp_eggs

    # Voronoi score: my_voronoi - opp_voronoi (already POV-relative)
    space_score = vor.vor_score

    # --- Phase terms ---
    moves_left  = cur_board.turns_left_player
    total_moves = cur_board.MAX_TURNS
    phase_mat   = max(0.0, min(1.0, moves_left / total_moves))  # 1 early, 0 late

    # Openness
    max_contested = 8.0
    openness = 0.0
    if max_contested > 0:
        openness = max(0.0, min(1.0, vor.contested / max_contested))

    # --- Fragmentation & frontier geometry ---
    frag_score = vor.frag_score
    frag_score = max(0.0, min(1.0, frag_score))
    
    # Gradient components
    grads = {}
    
    # w_space = W_SPACE_MIN + phase_mat * (W_SPACE_MAX - W_SPACE_MIN)
    # w_space = W_SPACE_MIN * (1 - phase_mat) + W_SPACE_MAX * phase_mat
    # space_term = w_space * space_score
    if openness > 0:
        grads["W_SPACE_MIN"] = (1.0 - phase_mat) * space_score
        grads["W_SPACE_MAX"] = phase_mat * space_score
    else:
        grads["W_SPACE_MIN"] = 0.0
        grads["W_SPACE_MAX"] = 0.0

    # w_mat = W_MAT_MIN + (1.0 - phase_mat) * (W_MAT_MAX - W_MAT_MIN)
    # w_mat = W_MAT_MIN * phase_mat + W_MAT_MAX * (1.0 - phase_mat)
    # mat_term = w_mat * mat_diff
    grads["W_MAT_MIN"] = phase_mat * mat_diff
    grads["W_MAT_MAX"] = (1.0 - phase_mat) * mat_diff

    # frag_term = -W_FRAG * openness * frag_score
    grads["W_FRAG"] = -1.0 * openness * frag_score

    # frontier_dist_term = -FRONTIER_COEFF * (1 - frag_score) * vor.average_contested_dist
    grads["FRONTIER_COEFF"] = -1.0 * (1 - frag_score) * vor.average_contested_dist
    
    # weighted_contested_term = W_WEIGHTED_CONTESTED * vor.weighted_contested
    grads["W_WEIGHTED_CONTESTED"] = vor.weighted_contested

    # W_BLOCKS = mat_term / 5
    # blocks_term = W_BLOCKS * (vor.my_owned - vor.opp_owned)
    # blocks_term = (w_mat * mat_diff / 5) * blocks_diff
    # This adds to the gradient of w_mat
    blocks_diff = vor.my_owned - vor.opp_owned
    
    # d(blocks_term)/d(w_mat) = (mat_diff / 5) * blocks_diff
    # d(blocks_term)/d(W_MAT_MIN) = d(blocks_term)/d(w_mat) * d(w_mat)/d(W_MAT_MIN)
    #                             = (mat_diff * blocks_diff / 5) * phase_mat
    grads["W_MAT_MIN"] += (mat_diff * blocks_diff / 5.0) * phase_mat
    grads["W_MAT_MAX"] += (mat_diff * blocks_diff / 5.0) * (1.0 - phase_mat)

    return grads


def get_board_string(board: board_mod.Board, trapdoors=set()):
    """
    Returns a string representation of the current state of the board.
    """

    main_list = []
    chicken_a = board.chicken_player if board.is_as_turn else board.chicken_enemy
    chicken_b = board.chicken_enemy if board.is_as_turn else board.chicken_player

    if board.is_as_turn:
        a_loc = board.chicken_player.get_location()
        a_eggs = board.eggs_player
        a_turds = board.turds_player
        b_loc = board.chicken_enemy.get_location()
        b_eggs = board.eggs_enemy
        b_turds = board.turds_enemy
    else:
        a_loc = board.chicken_enemy.get_location()
        a_eggs = board.eggs_enemy
        a_turds = board.turds_enemy
        b_loc = board.chicken_player.get_location()
        b_eggs = board.eggs_player
        b_turds = board.turds_player

    dim = board.game_map.MAP_SIZE
    main_list.append("  ")
    for x in range(dim):
        main_list.append(f"{x} ")
    main_list.append("\n")

    for y in range(dim):
        main_list.append(f"{y} ")
        for x in range(dim):
            current_loc = (x, y)
            if a_loc == current_loc:
                main_list.append("@ ")
            elif b_loc == current_loc:
                main_list.append("% ")
            elif current_loc in a_eggs:
                main_list.append("a ")
            elif current_loc in a_turds:
                main_list.append("A ")
            elif current_loc in b_eggs:
                main_list.append("b ")
            elif current_loc in b_turds:
                main_list.append("B ")
            elif current_loc in trapdoors:
                main_list.append("T ")
            else:
                main_list.append("  ")

        main_list.append("\n")

    return_string = "".join(main_list)
    return (
        return_string,
        chicken_a.get_eggs_laid(),
        chicken_b.get_eggs_laid(),
        chicken_a.get_turds_left(),
        chicken_b.get_turds_left(),
    )


def debug_evaluate(cur_board: board_mod.Board, vor: VoronoiInfo, trap_belief : TrapdoorBelief, known_traps: set) -> None:
    """
    Debug version of evaluate that prints components.
    """
    if cur_board.is_game_over():
        print("Game Over")
        return

    dim = cur_board.game_map.MAP_SIZE

    # --- Basic features: material and space ---

    my_eggs  = cur_board.chicken_player.eggs_laid
    opp_eggs = cur_board.chicken_enemy.eggs_laid
    mat_diff = my_eggs - opp_eggs

    # Voronoi score: my_voronoi - opp_voronoi (already POV-relative)
    space_score = vor.vor_score

    # --- Phase terms: time-based for material, structure-based for space/risk ---

    # Time-to-go: material matters more as we approach the end.
    moves_left  = cur_board.turns_left_player
    total_moves = cur_board.MAX_TURNS
    phase_mat   = max(0.0, min(1.0, moves_left / total_moves))  # 1 early, 0 late

    # Openness: how much frontier there is. 0 = closed, 1 = very open.
    # There can't be more than ~8 contested squares; normalize by that.
    max_contested = 8.0
    openness = 0.0
    if max_contested > 0:
        openness = max(0.0, min(1.0, vor.contested / max_contested))

    # --- Weights in egg units (tunable) ---
    # Using HeuristicWeights class

    # Interpolate weights
    w_space = HeuristicWeights.W_SPACE_MIN + phase_mat * (HeuristicWeights.W_SPACE_MAX - HeuristicWeights.W_SPACE_MIN)
    if openness == 0.00:
        w_space = 0.0
    w_mat   = HeuristicWeights.W_MAT_MIN   + (1.0 - phase_mat) * (HeuristicWeights.W_MAT_MAX - HeuristicWeights.W_MAT_MIN)

    # --- Fragmentation & frontier geometry ---
    # 1) Fragmentation score: 0 (all frontier in one blob) → 1 (highly fragmented).
    # Penalize more when the board is open.
    frag_score = vor.frag_score  # fail-safe if not set
    frag_score = max(0.0, min(1.0, frag_score))
    frag_term  = -HeuristicWeights.W_FRAG * openness * frag_score

    # 2) Max contested distance from *my chicken* to any contested square.
    # Encourage being close to the whole frontier: large distance = bad.
    frontier_dist_term = -HeuristicWeights.FRONTIER_COEFF * (1 - frag_score) * vor.average_contested_dist

    PANIC_THRESHOLD = HeuristicWeights.PANIC_THRESHOLD 
    
    CLOSEST_EGG_COEFF = (w_mat - 5) * 0.1 if moves_left <= PANIC_THRESHOLD or openness == 0 else 0.0
    egg_dist = vor.min_egg_dist if vor.min_egg_dist <= 63 else 0


    

    # --- Combine everything ---

    space_term = w_space * space_score
    mat_term   = w_mat   * mat_diff
    W_BLOCKS = mat_term / 5
    
    # Weighted contested term
    # Reward holding contested squares that lead to large regions
    weighted_contested_term = HeuristicWeights.W_WEIGHTED_CONTESTED * vor.weighted_contested

    score = space_term + mat_term + frag_term + frontier_dist_term - CLOSEST_EGG_COEFF * egg_dist * 0.25 + W_BLOCKS * (vor.my_owned - vor.opp_owned) + weighted_contested_term

    if cur_board.chicken_player.eggs_laid + (moves_left-vor.min_egg_dist) / 2 < cur_board.chicken_enemy.eggs_laid:
        score -= INF
        score += (my_eggs - opp_eggs) * 100

        
    # Print Debug Info
    
    # Print Debug Info
    
    # Helper to print grid
    def print_grid(title, grid_func, cell_width=1):
        lines = []
        lines.append(f"--- {title} ---")
        header = "  " + " ".join(f"{x:>{cell_width}}" for x in range(dim))
        lines.append(header)
        for y in range(dim):
            row = [f"{y}"]
            for x in range(dim):
                val = grid_func(x, y)
                row.append(f"{val:>{cell_width}}")
            lines.append(" ".join(row))
        return "\n".join(lines)

    output = []
    output.append("\n" + "="*40)
    output.append(f"DEBUG EVALUATION (Turn {cur_board.turn_count})")
    output.append(f"Moves Left: {moves_left}")
    output.append(f"Total Score: {score:.2f}")
    
    # 1. Standard Board
    board_str, _, _, _, _ = get_board_string(cur_board, known_traps)
    output.append(board_str)
    
    # 2. Owner Grid
    def owner_char(x, y):
        o = vor.owner[x][y]
        if o == OWNER_ME: return "M"
        if o == OWNER_OPP: return "O"
        return "."
    output.append(print_grid("Owner Grid", owner_char, 1))
    
    # 3. Contested Squares
    def contested_char(x, y):
        o = vor.owner[x][y]
        if o == OWNER_ME:
            # Check neighbors for opponent
            is_border = False
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < dim and 0 <= ny < dim:
                    if vor.owner[nx][ny] == OWNER_OPP:
                        is_border = True
            if is_border: return "C"
            return "."
        return " "
    output.append(print_grid("Approx Contested (Border)", contested_char, 1))

    # 4. Regions (Sizes)
    def region_char(x, y):
        s = vor.region_sizes[x][y]
        if s > 0: return f"{s}"
        return "."
    output.append(print_grid("Region Sizes", region_char, 2))

    # 5. Trap Belief
    def trap_char(x, y):
        p = trap_belief.prob_at((x, y))
        if p < 0.01: return "."
        if p > 0.99: return "100"
        return f"{int(p*100)}"
    output.append(print_grid("Trap Belief (%)", trap_char, 3))

    output.append("-" * 20)
    output.append(f"Material Term:      {mat_term:.2f} (Diff: {mat_diff}, W: {w_mat:.2f})")
    output.append(f"Space Term:         {space_term:.2f} (Score: {space_score}, W: {w_space:.2f})")
    output.append(f"Frag Term:          {frag_term:.2f} (Score: {frag_score:.2f})")
    output.append(f"Frontier Dist Term: {frontier_dist_term:.2f} (Avg Dist: {vor.average_contested_dist:.2f})")
    output.append(f"Weighted Contested: {weighted_contested_term:.2f} (Raw: {vor.weighted_contested:.2f})")
    
    # Sync W_BLOCKS with evaluate function
    W_BLOCKS_DEBUG = mat_term / 5
    output.append(f"Blocks Term:        {W_BLOCKS_DEBUG * (vor.my_owned - vor.opp_owned):.2f} (My: {vor.my_owned}, Opp: {vor.opp_owned})")
    
    output.append(f"Egg Dist Penalty:   {-CLOSEST_EGG_COEFF * egg_dist * 0.25:.2f}")
    output.append(f"Openness:           {openness:.2f}")
    output.append("="*40 + "\n")
    
    # Write to file
    with open("debug_log.txt", "a") as f:
        f.write("\n".join(output))

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
    # REMOVED: This caused a bug where we wouldn't consider trapping the opponent if we didn't own any border squares.
    # if total_contested == 0:
    #     filtered = [mv for mv in filtered if mv[1] != MoveType.TURD]

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
    # Using HeuristicWeights class

    # How much contested counts by direction matter
    # Using HeuristicWeights class

    # TURD buffs
    # Using HeuristicWeights class

    scored: List[tuple[float, Move]] = []

    for mv in filtered:
        direction, mtype = mv

        # 4a) Base by move type
        if mtype == MoveType.EGG:
            score = HeuristicWeights.BASE_EGG
        elif mtype == MoveType.PLAIN:
            score = HeuristicWeights.BASE_PLAIN
        else:  # MoveType.TURD
            score = HeuristicWeights.BASE_TURD

        # 4b) Direction: more contested squares in that direction → more priority
        dir_contested = contested_by_dir.get(direction, 0)
        score += HeuristicWeights.DIR_WEIGHT * dir_contested

        # 4c) Contextual TURD buffs
        if mtype == MoveType.TURD:
            if my_center_dist <= 2:
                # TURD > PLAIN when we control center
                score += HeuristicWeights.TURD_CENTER_BONUS
            if near_frontier:
                # TURD can beat EGG near frontier
                score += HeuristicWeights.TURD_FRONTIER_BONUS

        # Tiny jitter to break ties
        scored.append((score, mv))

    # ----------------------------
    # 5) Sort best → worst
    # ----------------------------

    scored.sort(key=lambda x: x[0], reverse=True)
    return [mv for _, mv in scored]

