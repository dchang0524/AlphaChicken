from __future__ import annotations
from collections import deque
from typing import Dict, List, Tuple, Set, Optional
from game import board as board_mod  # type: ignore

DIST_UNREACHABLE = -1
DIST_BLOCKED     = -2

OWNER_NONE = 0
OWNER_ME   = 1
OWNER_OPP  = 2

def bfs_single(
    cur_board: board_mod.Board,
    start: Tuple[int, int],
    opp_eggs: Set[Tuple[int, int]],
    opp_turds: Set[Tuple[int, int]],
    known_traps: Set[Tuple[int, int]],
) -> List[List[int]]:
    """
    Single-player BFS on ChickenFight board using ONE 2D distance array.

    Obstacles (BLOCKED):
      - opponent eggs
      - opponent turds
      - known trapdoors
      - squares adjacent (4-neighbor) to opponent turds

    Own eggs/turds are passable.
    No probabilistic 'lava' traps.
    """

    dim = cur_board.game_map.MAP_SIZE

    # Initialize all cells as unreachable (but not yet blocked)
    dist: List[List[int]] = [
        [DIST_UNREACHABLE for _ in range(dim)] for _ in range(dim)
    ]

    # 1) Mark hard obstacles directly in dist as BLOCKED
    for (x, y) in opp_eggs | opp_turds | set(known_traps):
        dist[x][y] = DIST_BLOCKED

    # 2) Mark adjacency around opponent turds as BLOCKED (turd aura)
    for (tx, ty) in opp_turds:
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = tx + dx, ty + dy
            if 0 <= nx < dim and 0 <= ny < dim:
                dist[nx][ny] = DIST_BLOCKED

    sx, sy = start

    # If starting square is blocked, nothing is reachable
    if dist[sx][sy] == DIST_BLOCKED:
        return dist

    # 3) Standard BFS using dist[][] only
    q: deque[Tuple[int, int]] = deque()
    dist[sx][sy] = 0
    q.append((sx, sy))

    while q:
        x, y = q.popleft()
        d = dist[x][y]

        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if not (0 <= nx < dim and 0 <= ny < dim):
                continue

            # Only step onto cells that are still UNREACHABLE.
            # BLOCKED cells (DIST_BLOCKED) and already-visited cells (>=0) are skipped.
            if dist[nx][ny] != DIST_UNREACHABLE:
                continue

            dist[nx][ny] = d + 1
            q.append((nx, ny))

    return dist


def bfs_distances_both(
    cur_board: board_mod.Board,
    known_traps: Set[Tuple[int, int]],
) -> Tuple[List[List[int]], List[List[int]]]:
    """
    Convenience wrapper:
      - Runs bfs_single for the current player ("me")
      - Runs bfs_single for the enemy ("opp")

    Returns:
      dist_me[x][y]  : steps from cur_board.chicken_player to (x,y), or <0 if unreachable/blocked
      dist_opp[x][y] : steps from cur_board.chicken_enemy to (x,y), or <0 if unreachable/blocked
    """

    me_chicken  = cur_board.chicken_player
    opp_chicken = cur_board.chicken_enemy

    eggs_me   = cur_board.eggs_player
    eggs_opp  = cur_board.eggs_enemy
    turds_me  = cur_board.turds_player
    turds_opp = cur_board.turds_enemy

    # Current player: blocked by opponent pieces + adjacent-to-opponent-turds + known_traps
    dist_me = bfs_single(
        cur_board=cur_board,
        start=me_chicken.get_location(),
        opp_eggs=eggs_opp,
        opp_turds=turds_opp,
        known_traps=known_traps,
    )

    # Opponent: blocked by *our* pieces + adjacent-to-our-turds + known_traps
    dist_opp = bfs_single(
        cur_board=cur_board,
        start=opp_chicken.get_location(),
        opp_eggs=eggs_me,
        opp_turds=turds_me,
        known_traps=known_traps,
    )

    return dist_me, dist_opp


def compute_region_sizes(
    cur_board: board_mod.Board,
    owner_grid: List[List[int]],
    dist_me: List[List[int]],
    dist_opp: List[List[int]],
) -> Dict[int, Dict[str, Any]]:
    """
    Compute connected safe regions and produce *only* region-level stats:
      - size
      - me_frac, opp_frac
      - min contested distance for ME
      - min contested distance for OPP

    A contested square is defined by:
        dist_me >= 0
        dist_opp >= 0
        abs(dist_me - dist_opp) <= 1

    Returns:
      region_stats[rid] = {
          "size": int,
          "me_frac": float,
          "opp_frac": float,
          "min_contested_dist_me": float | None,
          "min_contested_dist_opp": float | None,
      }
    """
    dim = cur_board.game_map.MAP_SIZE
    visited = [[False]*dim for _ in range(dim)]

    # --- Build blocked grid ---
    opp_eggs  = cur_board.eggs_enemy
    opp_turds = cur_board.turds_enemy
    my_eggs   = cur_board.eggs_player
    my_turds  = cur_board.turds_player

    blocked = [[False]*dim for _ in range(dim)]

    # Hard blocks
    for (x, y) in opp_eggs | opp_turds | my_eggs | my_turds:
        blocked[x][y] = True

    # Adjacency to turds also blocks
    for (tx, ty) in opp_turds | my_turds:
        for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
            nx, ny = tx+dx, ty+dy
            if 0 <= nx < dim and 0 <= ny < dim:
                blocked[nx][ny] = True

    region_stats: Dict[int, Dict[str, Any]] = {}
    rid = 1

    # --- Main BFS over all safe cells ---
    for x in range(dim):
        for y in range(dim):
            if visited[x][y] or blocked[x][y]:
                continue

            # Start BFS for a new region
            q = deque([(x, y)])
            visited[x][y] = True

            size = 0
            me_cells = 0
            opp_cells = 0

            min_me = float('inf')
            min_opp = float('inf')

            while q:
                cx, cy = q.popleft()
                size += 1

                owner = owner_grid[cx][cy]
                if owner == OWNER_ME:
                    me_cells += 1
                elif owner == OWNER_OPP:
                    opp_cells += 1

                # contested condition
                dme = dist_me[cx][cy]
                dop = dist_opp[cx][cy]
                if dme >= 0 and dop >= 0 and abs(dme - dop) <= 1:
                    if dme < min_me:
                        min_me = dme
                    if dop < min_opp:
                        min_opp = dop

                # expand BFS
                for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
                    nx, ny = cx+dx, cy+dy
                    if 0 <= nx < dim and 0 <= ny < dim:
                        if not visited[nx][ny] and not blocked[nx][ny]:
                            visited[nx][ny] = True
                            q.append((nx, ny))

            # convert infinities to None
            min_me_val = None if min_me == float('inf') else min_me
            min_opp_val = None if min_opp == float('inf') else min_opp

            region_stats[rid] = {
                "size": size,
                "me_cells": me_cells,
                "opp_cells": opp_cells,
                "me_frac": me_cells / size,
                "opp_frac": opp_cells / size,
                "min_contested_dist_me": min_me_val,
                "min_contested_dist_opp": min_opp_val,
            }

            rid += 1

    return region_stats

