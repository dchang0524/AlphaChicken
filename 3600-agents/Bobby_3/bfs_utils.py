from __future__ import annotations
from collections import deque
from typing import Tuple, Set, List
import numpy as np
from game import board as board_mod  # type: ignore

DIST_UNREACHABLE = -1
DIST_BLOCKED     = -2

def bfs_single(
    cur_board: board_mod.Board,
    start: Tuple[int, int],
    opp_eggs: Set[Tuple[int, int]],
    opp_turds: Set[Tuple[int, int]],
    known_traps: Set[Tuple[int, int]],
) -> np.ndarray:
    """
    Optimized BFS using NumPy and Sentinel Padding.
    Returns a NumPy array, NOT a list of lists.
    """
    dim = cur_board.game_map.MAP_SIZE
    
    # 1. Initialize Padded Grid (dim+2 x dim+2)
    padded_dist = np.full((dim + 2, dim + 2), DIST_UNREACHABLE, dtype=np.int8)

    # 2. Create Sentinel Border (Walls)
    padded_dist[0, :]  = DIST_BLOCKED
    padded_dist[-1, :] = DIST_BLOCKED
    padded_dist[:, 0]  = DIST_BLOCKED
    padded_dist[:, -1] = DIST_BLOCKED

    # 3. Mark Obstacles
    obstacles = []
    
    # Add Opponent Eggs
    for x, y in opp_eggs:
        obstacles.append((x + 1, y + 1))
        
    # Add Known Traps
    for x, y in known_traps:
        obstacles.append((x + 1, y + 1))

    # Add Opponent Turds + Turd Aura
    for tx, ty in opp_turds:
        px, py = tx + 1, ty + 1
        obstacles.append((px, py))
        obstacles.append((px + 1, py))
        obstacles.append((px - 1, py))
        obstacles.append((px, py + 1))
        obstacles.append((px, py - 1))

    # Bulk update blocking
    if obstacles:
        rows, cols = zip(*obstacles)
        padded_dist[rows, cols] = DIST_BLOCKED

    # 4. BFS Initialization
    sx, sy = start
    start_padded = (sx + 1, sy + 1)
    
    if padded_dist[start_padded] == DIST_BLOCKED:
        return padded_dist[1:-1, 1:-1]

    padded_dist[start_padded] = 0
    q = deque([start_padded])

    # 5. The Hot Loop
    while q:
        cx, cy = q.popleft()
        next_dist = padded_dist[cx, cy] + 1

        # Unrolled neighbor checks
        nx, ny = cx + 1, cy
        if padded_dist[nx, ny] == DIST_UNREACHABLE:
            padded_dist[nx, ny] = next_dist
            q.append((nx, ny))

        nx, ny = cx - 1, cy
        if padded_dist[nx, ny] == DIST_UNREACHABLE:
            padded_dist[nx, ny] = next_dist
            q.append((nx, ny))

        nx, ny = cx, cy + 1
        if padded_dist[nx, ny] == DIST_UNREACHABLE:
            padded_dist[nx, ny] = next_dist
            q.append((nx, ny))

        nx, ny = cx, cy - 1
        if padded_dist[nx, ny] == DIST_UNREACHABLE:
            padded_dist[nx, ny] = next_dist
            q.append((nx, ny))

    return padded_dist[1:-1, 1:-1]


def bfs_distances_both(
    cur_board: board_mod.Board,
    known_traps: Set[Tuple[int, int]],
) -> Tuple[np.ndarray, np.ndarray]:
    
    me_chicken  = cur_board.chicken_player
    opp_chicken = cur_board.chicken_enemy

    eggs_me   = cur_board.eggs_player
    eggs_opp  = cur_board.eggs_enemy
    turds_me  = cur_board.turds_player
    turds_opp = cur_board.turds_enemy

    dist_me = bfs_single(
        cur_board=cur_board,
        start=me_chicken.get_location(),
        opp_eggs=eggs_opp,
        opp_turds=turds_opp,
        known_traps=known_traps,
    )

    dist_opp = bfs_single(
        cur_board=cur_board,
        start=opp_chicken.get_location(),
        opp_eggs=eggs_me,
        opp_turds=turds_me,
        known_traps=known_traps,
    )

    return dist_me, dist_opp