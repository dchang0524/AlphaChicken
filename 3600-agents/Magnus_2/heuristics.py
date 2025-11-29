
import math
from typing import List, Tuple, Dict, Deque
from collections import deque

from game import board as board_mod  # type: ignore
from .voronoi import analyze, VoronoiInfo, OWNER_NONE, OWNER_ME, OWNER_OPP  # adjust import path as needed
from .hiddenMarkov import TrapdoorBelief
from game.enums import Direction, MoveType  # type: ignore
from .weights import HeuristicWeights
from .bfs_utils import bfs_distances_both

INF = HeuristicWeights.W_LOSS_PENALTY #INF shouldn't be that big since we don't want it to mess up expectimax


def evaluate(cur_board: board_mod.Board, vor: VoronoiInfo, trap_belief : TrapdoorBelief) -> float:
    """
    Magnus evaluation function.
    Score = (My Score) - (Opponent Score)
    """
    
    # --- SHARED DATA ---
    dim = cur_board.game_map.MAP_SIZE
    my_eggs  = cur_board.chicken_player.eggs_laid
    opp_eggs = cur_board.chicken_enemy.eggs_laid
    
    # Game Over Check
    if cur_board.is_game_over():
        if my_eggs > opp_eggs:
            return INF + (my_eggs - opp_eggs)
        elif my_eggs < opp_eggs:
            return -INF + (my_eggs - opp_eggs)
        else:
            return 0.0


    # --- Phase / Progression ---
    moves_left  = cur_board.turns_left_player
    cap = moves_left / 2.0 + 2.0
    total_moves = cur_board.MAX_TURNS
    phase_mat   = max(0.0, min(1.0, moves_left / total_moves))  # 1 early, 0 late
    
    # --- MY SCORE CALCULATION ---
    
    # 1. Base Egg (Difference handled at end, but we can track components)
    # We'll just use raw counts here and subtract later
    
    # 2. Potential Eggs
    my_safe_term = HeuristicWeights.W_SAFE_EGG * vor.my_safe_potential_eggs
    my_vor_term  = HeuristicWeights.W_VORONOI_EGG * vor.my_reachable_potential_eggs
    my_potential = min(my_safe_term + my_vor_term, cap)
    
    # 4. Egg Dist (Penalty), to nudge player towards eggs if the egg is too far for depth to pick it up
    my_egg_dist_score = 0.0
    if vor.min_egg_dist < dim * dim and vor.min_egg_dist > 3:
        my_egg_dist_score = -HeuristicWeights.W_EGG_DIST_PENALTY * (vor.min_egg_dist)
        
    # 5. Bad Turd (Penalty)
    my_bad_turd = -HeuristicWeights.W_BAD_TURD * vor.bad_turd_count
    
    # 6. Turd Savings
    my_turd_save = HeuristicWeights.W_TURD_SAVINGS * cur_board.chicken_player.get_turds_left()
    
    # 9. Contested Dist (Penalty)
    #my_contested = -HeuristicWeights.W_CONTESTED_SIG * vor.sum_weighted_contested_dist * (1.0 - phase_mat)
    my_contested = -HeuristicWeights.W_CONTESTED_SIG * vor.sum_weighted_contested_dist * ((1-HeuristicWeights.CONTESTED_OPENNESS_CORRELATION) + (1 - phase_mat) * HeuristicWeights.CONTESTED_OPENNESS_CORRELATION)
    
    
    # 12. Loss Prevention (Penalty)
    my_loss = 0.0
    my_theo_max = my_eggs + 4 + (moves_left - vor.min_egg_dist) / 2.0
    if my_theo_max < opp_eggs:
        my_loss = -HeuristicWeights.W_LOSS_PENALTY
        
    my_total = (
        my_eggs + # Base
        my_potential + 
        my_egg_dist_score + 
        my_turd_save + 
        my_bad_turd + 
        my_contested + 
        my_loss
    )
    
    # --- OPPONENT SCORE CALCULATION ---
    
    # 2. Potential Eggs
    opp_safe_term = HeuristicWeights.W_SAFE_EGG * vor.opp_safe_potential_eggs
    opp_vor_term  = HeuristicWeights.W_VORONOI_EGG * vor.opp_reachable_potential_eggs
    opp_potential = min(opp_safe_term + opp_vor_term, cap)
    
    # 4. Egg Dist (NOT RELATIVIZED - Absolute for me only)
    # opp_egg_dist_score = 0.0 
        
    # 5. Bad Turd
    opp_bad_turd = -HeuristicWeights.W_BAD_TURD * vor.opp_bad_turd_count
    
    # 6. Turd Savings
    opp_turd_save = HeuristicWeights.W_TURD_SAVINGS * cur_board.chicken_enemy.get_turds_left()
    
    # 9. Contested Dist
    #opp_contested = -HeuristicWeights.W_CONTESTED_SIG * vor.opp_sum_weighted_contested_dist * (1.0 - phase_mat)
    opp_contested = -HeuristicWeights.W_CONTESTED_SIG * vor.opp_sum_weighted_contested_dist * ((1-HeuristicWeights.CONTESTED_OPENNESS_CORRELATION) + (1 - phase_mat) * HeuristicWeights.CONTESTED_OPENNESS_CORRELATION)
    
    # 12. Loss Prevention
    opp_loss = 0.0
    opp_theo_max = opp_eggs + 4 + (moves_left - vor.opp_min_egg_dist) / 2.0
    if opp_theo_max < my_eggs:
        opp_loss = -HeuristicWeights.W_LOSS_PENALTY
        
    opp_total = (
        opp_eggs + # Base
        opp_potential + 
        opp_turd_save + 
        opp_bad_turd + 
        opp_contested + 
        opp_loss
    )

    return my_total - opp_total


def get_weight_gradient(cur_board: board_mod.Board, vor: VoronoiInfo, trap_belief : TrapdoorBelief) -> Dict[str, float]:
    """
    Gradient of Magnus evaluation.
    """
    if cur_board.is_game_over():
        return {}

    grads = {}
    
    # Base Egg (Implicit 1.0)
    
    # Potential Eggs
    # W_SAFE_EGG * min(safe, cap)
    moves_left  = cur_board.turns_left_player
    safe_egg_cap = moves_left / 2.0 + 2.0
    grads["W_SAFE_EGG"] = min(vor.my_safe_potential_eggs, safe_egg_cap)
    
    # W_VORONOI_EGG * reachable
    grads["W_VORONOI_EGG"] = vor.my_reachable_potential_eggs
    
    # Egg Dist Bonus
    dim = cur_board.game_map.MAP_SIZE
    if vor.min_egg_dist < dim * dim:
        grads["W_EGG_DIST_BONUS"] = 1.0 / (vor.min_egg_dist + 1.0)
    else:
        grads["W_EGG_DIST_BONUS"] = 0.0
        
    # Turd Savings
    grads["W_TURD_SAVINGS"] = cur_board.chicken_player.get_turds_left()
    
    # Contested Dist Penalty
    # -W_CONTESTED_SIG * sum * (1 - phase)
    total_moves = cur_board.MAX_TURNS
    phase_mat   = max(0.0, min(1.0, moves_left / total_moves))
    grads["W_CONTESTED_SIG"] = -vor.sum_weighted_contested_dist * (1.0 - phase_mat)
    
    # Loss Penalty
    my_eggs  = cur_board.chicken_player.eggs_laid
    opp_eggs = cur_board.chicken_enemy.eggs_laid
    theoretical_max = my_eggs + 4 + (moves_left - vor.min_egg_dist) / 2.0
    if theoretical_max < opp_eggs:
        grads["W_LOSS_PENALTY"] = -1.0
    else:
        grads["W_LOSS_PENALTY"] = 0.0

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



def debug_evaluate(
    cur_board: board_mod.Board,
    vor: VoronoiInfo,
    trap_belief: TrapdoorBelief,
    known_traps: set[Tuple[int, int]],
    my_risk: float = 0.0,
    opp_risk: float = 0.0,
) -> None:
    """
    Debug version of evaluate that prints components for BOTH players.
    Recomputes local owner grid and my region sizes instead of using vor.owner / vor.region_sizes.
    """
    dim = cur_board.game_map.MAP_SIZE

    # ------------------------------------------------------------
    # RECOMPUTE DISTANCES, OWNER GRID, AND REGION SIZES (MY)
    # ------------------------------------------------------------

    # 1) Distances
    dist_me, dist_opp = bfs_distances_both(cur_board, known_traps)

    # 2) Owner grid (same rule as in analyze)
    owner = [[OWNER_NONE for _ in range(dim)] for _ in range(dim)]
    for x in range(dim):
        for y in range(dim):
            d_me = dist_me[x][y]
            d_opp = dist_opp[x][y]

            reachable_me = (d_me >= 0)
            reachable_opp = (d_opp >= 0)

            if reachable_me and (not reachable_opp or d_me <= d_opp):
                owner[x][y] = OWNER_ME
            elif reachable_opp:
                owner[x][y] = OWNER_OPP
            # else OWNER_NONE

    # 3) Region sizes for MY connected safe regions
    region_sizes_my = [[0 for _ in range(dim)] for _ in range(dim)]
    visited = [[False for _ in range(dim)] for _ in range(dim)]

    eggs_me = cur_board.eggs_player
    eggs_opp = cur_board.eggs_enemy
    turds_me = cur_board.turds_player
    turds_opp = cur_board.turds_enemy

    # Build blocked grid using same rules as compute_region_sizes
    blocked = [[False for _ in range(dim)] for _ in range(dim)]

    # Hard blocks: all eggs and turds
    for (x, y) in eggs_me | eggs_opp | turds_me | turds_opp:
        blocked[x][y] = True

    # Adjacent to any turd is also blocked
    for (tx, ty) in turds_me | turds_opp:
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = tx + dx, ty + dy
            if 0 <= nx < dim and 0 <= ny < dim:
                blocked[nx][ny] = True

    # Flood-fill only over MY owned, unblocked cells
    for x in range(dim):
        for y in range(dim):
            if (
                not visited[x][y]
                and not blocked[x][y]
                and owner[x][y] == OWNER_ME
            ):
                q = deque([(x, y)])
                visited[x][y] = True
                component = [(x, y)]

                while q:
                    cx, cy = q.popleft()
                    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nx, ny = cx + dx, cy + dy
                        if 0 <= nx < dim and 0 <= ny < dim:
                            if (
                                not visited[nx][ny]
                                and not blocked[nx][ny]
                                and owner[nx][ny] == OWNER_ME
                            ):
                                visited[nx][ny] = True
                                q.append((nx, ny))
                                component.append((nx, ny))

                size = len(component)
                for (cx, cy) in component:
                    region_sizes_my[cx][cy] = size

    # ------------------------------------------------------------
    # ORIGINAL SCORING LOGIC (UNCHANGED)
    # ------------------------------------------------------------
    
    my_eggs  = cur_board.chicken_player.eggs_laid
    opp_eggs = cur_board.chicken_enemy.eggs_laid
    
    my_moves_left  = cur_board.turns_left_player
    opp_moves_left = cur_board.turns_left_enemy  # if this doesn't exist, you were already approximating
    
    total_moves = cur_board.MAX_TURNS
    
    # --- MY SCORE CALCULATION ---
    
    # 1. Base Egg
    my_base_egg = (my_eggs)
    
    # 2. Potential Eggs
    my_cap = my_moves_left / 2.0
    my_safe_term = HeuristicWeights.W_SAFE_EGG * vor.my_safe_potential_eggs
    my_vor_term  = HeuristicWeights.W_VORONOI_EGG * vor.my_reachable_potential_eggs
    my_potential = min(my_safe_term + my_vor_term, my_cap)
    
    # 4. Egg Dist
    my_egg_dist_score = 0.0
    if vor.min_egg_dist < dim * dim and vor.min_egg_dist > 3:
        my_egg_dist_score = -HeuristicWeights.W_EGG_DIST_PENALTY * (vor.min_egg_dist)
        
    # 5. Bad Turd
    my_bad_turd = -HeuristicWeights.W_BAD_TURD * vor.bad_turd_count
    
    # 6. Turd Savings
    my_turd_save = HeuristicWeights.W_TURD_SAVINGS * cur_board.chicken_player.get_turds_left()
    
    # 9. Contested Dist
    my_phase = max(0.0, min(1.0, my_moves_left / total_moves))
    my_contested = -HeuristicWeights.W_CONTESTED_SIG * vor.sum_weighted_contested_dist * (1.0 - my_phase)
    
    # 12. Loss Prevention
    my_loss = 0.0
    my_theo_max = my_eggs + 4 + (my_moves_left - vor.min_egg_dist) / 2.0
    if my_theo_max < opp_eggs:
        my_loss = -HeuristicWeights.W_LOSS_PENALTY
        
    my_total = (
        my_base_egg + my_potential + my_egg_dist_score + 
        my_turd_save + my_bad_turd + my_contested + my_loss
    )
    
    # --- OPPONENT SCORE CALCULATION ---
    
    # 1. Base Egg
    opp_base_egg = (opp_eggs)
    
    # 2. Potential Eggs
    opp_cap = my_moves_left / 2.0 
    opp_safe_term = HeuristicWeights.W_SAFE_EGG * vor.opp_safe_potential_eggs
    opp_vor_term  = HeuristicWeights.W_VORONOI_EGG * vor.opp_reachable_potential_eggs
    opp_potential = min(opp_safe_term + opp_vor_term, opp_cap)
    
    # 5. Bad Turd
    opp_bad_turd = -HeuristicWeights.W_BAD_TURD * vor.opp_bad_turd_count
    
    # 6. Turd Savings
    opp_turd_save = HeuristicWeights.W_TURD_SAVINGS * cur_board.chicken_enemy.get_turds_left()
    
    # 9. Contested Dist
    opp_phase = my_phase  # approx
    opp_contested = -HeuristicWeights.W_CONTESTED_SIG * vor.opp_sum_weighted_contested_dist * (1.0 - opp_phase)
    
    # 12. Loss Prevention
    opp_loss = 0.0
    opp_theo_max = opp_eggs + 4 + (my_moves_left - vor.opp_min_egg_dist) / 2.0
    if opp_theo_max < my_eggs:
        opp_loss = -HeuristicWeights.W_LOSS_PENALTY
        
    opp_total = (
        opp_base_egg + opp_potential + 
        opp_turd_save + opp_bad_turd + opp_contested + opp_loss
    )

    # ------------------------------------------------------------
    # OUTPUT (UNCHANGED FORMATTING, JUST USING NEW OWNER/REGIONS)
    # ------------------------------------------------------------
    
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
    output.append("\n" + "="*60)
    output.append(f"DEBUG EVALUATION (Turn {cur_board.turn_count})")
    output.append(f"{'METRIC':<20} | {'MY SCORE':>10} | {'OPP SCORE':>10}")
    output.append("-" * 46)
    output.append(f"{'Base Egg':<20} | {my_base_egg:>10.2f} | {opp_base_egg:>10.2f}")
    output.append(f"{'Potential Eggs':<20} | {my_potential:>10.2f} | {opp_potential:>10.2f}")
    output.append(f"{'  (Safe)':<20} | {vor.my_safe_potential_eggs:>10.1f} | {vor.opp_safe_potential_eggs:>10.1f}")
    output.append(f"{'  (Reachable)':<20} | {vor.my_reachable_potential_eggs:>10.1f} | {vor.opp_reachable_potential_eggs:>10.1f}")
    output.append(f"{'Egg Dist':<20} | {my_egg_dist_score:>10.2f} | ")
    output.append(f"{'  (Min Dist)':<20} | {vor.min_egg_dist:>10d} | ")
    output.append(f"{'Bad Turd':<20} | {my_bad_turd:>10.2f} | {opp_bad_turd:>10.2f}")
    output.append(f"{'Turd Savings':<20} | {my_turd_save:>10.2f} | {opp_turd_save:>10.2f}")
    output.append(f"{'Contested Dist':<20} | {my_contested:>10.2f} | {opp_contested:>10.2f}")
    output.append(f"{'  (Sum W. Dist)':<20} | {vor.sum_weighted_contested_dist:>10.2f} | {vor.opp_sum_weighted_contested_dist:>10.2f}")
    output.append(f"{'Risk Penalty':<20} | {my_risk:>10.2f} | {opp_risk:>10.2f}")
    output.append(f"{'Loss Penalty':<20} | {my_loss:>10.2f} | {opp_loss:>10.2f}")
    output.append("-" * 46)
    output.append("-" * 46)
    # Add risk to totals for display (since negamax does it)
    my_total_with_risk = my_total + my_risk
    opp_total_with_risk = opp_total + opp_risk
    output.append(f"{'TOTAL':<20} | {my_total_with_risk:>10.2f} | {opp_total_with_risk:>10.2f}")
    
    output.append("\n")
    
    # 1. Standard Board
    board_str, _, _, _, _ = get_board_string(cur_board, known_traps)
    output.append(board_str)
    
    # 2. Owner Grid (using recomputed owner)
    def owner_char(x, y):
        o = owner[x][y]
        if o == OWNER_ME: return "M"
        if o == OWNER_OPP: return "O"
        return "."
    output.append(print_grid("Owner Grid", owner_char, 1))
    
    # 3. Region Sizes (My) using recomputed region_sizes_my
    def region_char(x, y):
        s = region_sizes_my[x][y]
        if s > 0: return f"{s}"
        return "."
    output.append(print_grid("Region Sizes (My)", region_char, 2))

    # 4. Trap Belief (%)
    def trap_char(x, y):
        p = trap_belief.prob_at((x, y))
        if p < 0.01: return "."
        if p > 0.99: return "100"
        return f"{int(p*100)}"
    output.append(print_grid("Trap Belief (%)", trap_char, 3))

    output.append("="*60 + "\n")
    
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
    Order moves using HeuristicWeights.
    """
    if not moves:
        return moves

    dim = cur_board.game_map.MAP_SIZE
    filtered = list(moves)
    
    # Basic filtering
    # If there are EGG moves, filter out PLAIN moves
    egg_moves = [mv for mv in filtered if mv[1] == MoveType.EGG]
    if egg_moves:
        filtered = [mv for mv in filtered if mv[1] != MoveType.PLAIN]
    
    if not filtered:
        filtered = list(moves)

    contested_by_dir: dict[Direction, int] = {
        Direction.LEFT:  vor.contested_left,
        Direction.RIGHT: vor.contested_right,
        Direction.UP:    vor.contested_up,
        Direction.DOWN:  vor.contested_down,
    }

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

    scored: List[tuple[float, Move]] = []

    for mv in filtered:
        direction, mtype = mv
        if mtype == MoveType.EGG:
            score = HeuristicWeights.BASE_EGG
        elif mtype == MoveType.PLAIN:
            score = HeuristicWeights.BASE_PLAIN
        else:  # MoveType.TURD
            score = HeuristicWeights.BASE_TURD

        dir_contested = contested_by_dir.get(direction, 0)
        score += HeuristicWeights.DIR_WEIGHT * dir_contested

        if mtype == MoveType.TURD:
            if my_center_dist <= 2:
                score += HeuristicWeights.TURD_CENTER_BONUS
            if near_frontier:
                score += HeuristicWeights.TURD_FRONTIER_BONUS

        scored.append((score, mv))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [mv for _, mv in scored]
