from __future__ import annotations
import heapq
from typing import List, Tuple, Set
from game import board as board_mod  # type: ignore
from .hiddenMarkov import TrapdoorBelief  # Needed for probability lookup

DIST_UNREACHABLE = 999999.0
DIST_BLOCKED     = 999999.0

# How much extra "movement cost" a 100% trap adds.
# A 50% trap adds 0.5 * 10.0 = 5.0 cost.
TRAP_PENALTY_FACTOR = 10.0 

def bfs_single(
    cur_board: board_mod.Board,
    start: Tuple[int, int],
    opp_eggs: Set[Tuple[int, int]],
    opp_turds: Set[Tuple[int, int]],
    known_traps: Set[Tuple[int, int]],
    trap_belief: TrapdoorBelief,
) -> List[List[float]]:
    """
    Weighted BFS (Dijkstra) that treats probabilistic traps as high-cost terrain.
    Returns a 2D grid of FLOATS.
    """

    dim = cur_board.game_map.MAP_SIZE

    # Initialize grid with Infinity
    dist: List[List[float]] = [
        [DIST_UNREACHABLE for _ in range(dim)] for _ in range(dim)
    ]

    # 1. Mark Hard Obstacles (Walls)
    # We use the same 'unreachable' value effectively as a wall, 
    # but logically we skip them during expansion.
    blocked_cells = set(opp_eggs) | set(opp_turds) | known_traps
    
    # 2. Mark Turd Aura
    for (tx, ty) in opp_turds:
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = tx + dx, ty + dy
            if 0 <= nx < dim and 0 <= ny < dim:
                blocked_cells.add((nx, ny))

    sx, sy = start
    
    # If start is blocked (shouldn't happen usually), return empty
    if (sx, sy) in blocked_cells:
        return dist

    # 3. Dijkstra Initialization
    # Priority Queue stores: (current_cost, x, y)
    pq = [(0.0, sx, sy)]
    dist[sx][sy] = 0.0

    while pq:
        d, x, y = heapq.heappop(pq)

        # Lazy Deletion: If we found a cheaper way to this node previously, skip
        if d > dist[x][y]:
            continue

        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy

            # Bounds check
            if not (0 <= nx < dim and 0 <= ny < dim):
                continue

            # Hard Block check
            if (nx, ny) in blocked_cells:
                continue

            # --- LAVA COST CALCULATION ---
            # Base move cost is 1.0
            move_cost = 1.0
            
            # Add penalty based on trap probability
            # We look up the probability that a trap is HERE at (nx, ny)
            prob = trap_belief.prob_at((nx, ny))
            
            if prob > 0.0:
                move_cost += prob * TRAP_PENALTY_FACTOR

            new_dist = d + move_cost

            # Relaxation step
            if new_dist < dist[nx][ny]:
                dist[nx][ny] = new_dist
                heapq.heappush(pq, (new_dist, nx, ny))

    return dist


def bfs_distances_both(
    cur_board: board_mod.Board,
    known_traps: Set[Tuple[int, int]],
    trap_belief: TrapdoorBelief,
) -> Tuple[List[List[float]], List[List[float]]]:
    """
    Wrapper for both players.
    Note: Opponent also "fears" traps in our calculation, assuming they play optimally.
    """

    me_chicken  = cur_board.chicken_player
    opp_chicken = cur_board.chicken_enemy

    eggs_me   = cur_board.eggs_player
    eggs_opp  = cur_board.eggs_enemy
    turds_me  = cur_board.turds_player
    turds_opp = cur_board.turds_enemy

    # My distances (My obstacles are Opponent stuff)
    dist_me = bfs_single(
        cur_board=cur_board,
        start=me_chicken.get_location(),
        opp_eggs=eggs_opp,
        opp_turds=turds_opp,
        known_traps=known_traps,
        trap_belief=trap_belief
    )

    # Opponent distances (Opponent obstacles are My stuff)
    dist_opp = bfs_single(
        cur_board=cur_board,
        start=opp_chicken.get_location(),
        opp_eggs=eggs_me,
        opp_turds=turds_me,
        known_traps=known_traps,
        trap_belief=trap_belief
    )

    return dist_me, dist_opp