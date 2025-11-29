from __future__ import annotations

from typing import List, Tuple, Set
from collections import deque
from game import board as board_mod  # type: ignore
from .bfs_utils import bfs_distances_both, compute_region_sizes 

OWNER_NONE = 0
OWNER_ME   = 1  # current player (board.chicken_player)
OWNER_OPP  = 2  # enemy (board.chicken_enemy)

def analyze(
    self,
    cur_board: board_mod.Board,
    known_traps: Set[Tuple[int, int]],
) -> VoronoiInfo:

    dim  = cur_board.game_map.MAP_SIZE
    last = dim - 1

    # 1) BFS distances (movement blocking handled inside BFS)
    dist_me, dist_opp = bfs_distances_both(cur_board, known_traps)

    # Parities
    my_even  = cur_board.chicken_player.even_chicken
    opp_even = cur_board.chicken_enemy.even_chicken

    # My current position (for directional contested counts)
    my_x, my_y = cur_board.chicken_player.get_location()

    # 2D owner grid
    owner: List[List[int]] = [
        [OWNER_NONE for _ in range(dim)] for _ in range(dim)
    ]

    my_owned   = 0 # of "blocks"
    opp_owned  = 0
    contested  = 0
    max_contested_dist = 0
    min_contested_dist = dim * dim
    average_contested_dist = 0
    min_egg_dist       = dim * dim

    my_voronoi  = 0
    opp_voronoi = 0

    # Directional frontier pressure
    contested_up = 0
    contested_right = 0
    contested_down = 0
    contested_left = 0
    
    # New metrics for Magnus
    my_reachable_potential_eggs = 0
    my_safe_potential_eggs = 0
    sum_weighted_contested_dist = 0.0
    bad_turd_count = 0
    
    # Opponent metrics
    opp_reachable_potential_eggs = 0
    opp_safe_potential_eggs = 0
    opp_sum_weighted_contested_dist = 0.0
    opp_bad_turd_count = 0
    opp_min_egg_dist = dim * dim
    
    # Pre-check occupancy to identify empty squares
    eggs_me   = cur_board.eggs_player
    eggs_opp  = cur_board.eggs_enemy
    turds_me  = cur_board.turds_player
    turds_opp = cur_board.turds_enemy
    
    def is_empty(x, y):
        if (x, y) in eggs_me: return False
        if (x, y) in eggs_opp: return False
        if (x, y) in turds_me: return False
        if (x, y) in turds_opp: return False
        if (x, y) in known_traps: return False
        return True
    
    # Per-player unclaimed egg weights
    total_unclaimed_eggs_me  = 0.0
    total_unclaimed_eggs_opp = 0.0

    for x in range(dim):
        for y in range(dim):

            d_me  = dist_me[x][y]
            d_opp = dist_opp[x][y]

            reachable_me  = (d_me  >= 0)
            reachable_opp = (d_opp >= 0)
            
            # Corner weight for potential eggs
            is_corner = (x == 0 or x == last) and (y == 0 or y == last)
            w = 3 if is_corner else 1

            # --------------------------
            # 1) Reachability ownership
            # --------------------------
            if reachable_me and (not reachable_opp or d_me <= d_opp):
                owner[x][y] = OWNER_ME
            elif reachable_opp:
                owner[x][y] = OWNER_OPP
            elif reachable_me and not reachable_opp:
                my_owned += 1

            # Count potential eggs (empty squares) and per-player totals
            if is_empty(x, y):
                #TODO: Check if its better to make the owenership check or not for potential eggs
                if owner[x][y] == OWNER_ME:
                    total_unclaimed_eggs_me += w
                elif owner[x][y] == OWNER_OPP:
                    total_unclaimed_eggs_opp += w

                # MY POTENTIAL EGGS
                if owner[x][y] == OWNER_ME:
                    square_even = ((x + y) & 1) == 0
                    min_egg_dist = min(min_egg_dist, d_me)
                    if square_even == my_even:
                        if reachable_opp:
                            my_reachable_potential_eggs += w
                        else:
                            my_safe_potential_eggs += w
                
                # OPP POTENTIAL EGGS
                elif owner[x][y] == OWNER_OPP:
                    square_even = ((x + y) & 1) == 0
                    if square_even == opp_even:
                        opp_min_egg_dist = min(opp_min_egg_dist, d_opp)
                        if reachable_me:
                            opp_reachable_potential_eggs += w
                        else:
                            opp_safe_potential_eggs += w

            # --------------------------
            # 2) Contested frontier (per-square stats)
            # --------------------------
            if (
                reachable_me
                and reachable_opp
                and owner[x][y] == OWNER_ME
                and abs(d_me - d_opp) <= 1
            ):
                contested += 1
                average_contested_dist += d_me
                if d_me > max_contested_dist:
                    max_contested_dist = d_me
                elif d_me < min_contested_dist:
                    min_contested_dist = d_me

                dx = x - my_x
                dy = y - my_y

                if dx != 0 or dy != 0:
                    if dx > 0:
                        contested_right += 1
                    else:
                        contested_left += 1
                if dy > 0:
                    contested_down += 1
                elif dy < 0:
                    contested_up += 1

            # --------------------------
            # 3) Parity-based Voronoi (corner-weighted)
            # --------------------------
            weight = w
            square_even = ((x + y) & 1) == 0
            is_my_par   = (square_even == my_even)
            is_opp_par  = (square_even == opp_even)

            if is_my_par and owner[x][y] == OWNER_ME:
                my_voronoi += weight

            if is_opp_par and owner[x][y] == OWNER_OPP:
                opp_voronoi += weight

    average_contested_dist /= contested if contested > 0 else 64

    # --------------------------
    # 5) Region Size Heuristic (using me_cells / opp_cells)
    # --------------------------
    if total_unclaimed_eggs_me <= 0:
        total_unclaimed_eggs_me = 1.0
    if total_unclaimed_eggs_opp <= 0:
        total_unclaimed_eggs_opp = 1.0

    # compute region-level stats once
    region_stats = compute_region_sizes(
        cur_board,
        owner_grid=owner,
        dist_me=dist_me,
        dist_opp=dist_opp,
    )

    sum_weighted_contested_dist = 0.0
    opp_sum_weighted_contested_dist = 0.0

    me_region_count = 0
    opp_region_count = 0

    for rid, stats in region_stats.items():
        size       = stats["size"]
        me_cells   = stats["me_cells"]
        opp_cells  = stats["opp_cells"]
        d_me_min   = stats["min_contested_dist_me"]
        d_opp_min  = stats["min_contested_dist_opp"]

        # --- My weighted contested distance ---
        # significance_me ~ "how much of *my* territory (cells) in this region
        # relative to my total unclaimed egg potential"
        if d_me_min is not None and me_cells > 0:
            significance_me = me_cells / total_unclaimed_eggs_me
            sum_weighted_contested_dist += significance_me * d_me_min
            me_region_count += 1

        # --- Opponent weighted contested distance ---
        if d_opp_min is not None and opp_cells > 0:
            significance_opp = opp_cells / total_unclaimed_eggs_opp
            opp_sum_weighted_contested_dist += significance_opp * d_opp_min
            opp_region_count += 1

    if me_region_count > 0:
        sum_weighted_contested_dist /= me_region_count
    else:
        sum_weighted_contested_dist = 0.0

    if opp_region_count > 0:
        opp_sum_weighted_contested_dist /= opp_region_count
    else:
        opp_sum_weighted_contested_dist = 0.0

    # --------------------------
    # 6) Bad Turd Calculation
    # --------------------------
    for tx, ty in turds_me:
        is_useless = True
        for dx, dy in [(0, 2), (0, -2), (2, 0), (-2, 0), (1,1), (1, -1), (-1, 1), (-1, -1)]:
            nx, ny = tx + dx, ty + dy
            if 0 <= nx < dim and 0 <= ny < dim:
                if dist_opp[nx][ny] >= 0:
                    is_useless = False
                    break
        
        if is_useless:
            bad_turd_count += 1
            
    for tx, ty in turds_opp:
        is_useless = True
        for dx, dy in [(0, 2), (0, -2), (2, 0), (-2, 0), (1,1), (1, -1), (-1, 1), (-1, -1)]:
            nx, ny = tx + dx, ty + dy
            if 0 <= nx < dim and 0 <= ny < dim:
                if dist_me[nx][ny] >= 0:
                    is_useless = False
                    break
        
        if is_useless:
            opp_bad_turd_count += 1

    return VoronoiInfo(
        my_owned           = my_owned,
        opp_owned          = opp_owned,
        contested          = contested,
        average_contested_dist = average_contested_dist,
        max_contested_dist = max_contested_dist,
        min_contested_dist = min_contested_dist,
        min_egg_dist       = min_egg_dist,
        my_voronoi         = my_voronoi,
        opp_voronoi        = opp_voronoi,
        contested_up       = contested_up,
        contested_right    = contested_right,
        contested_down     = contested_down,
        contested_left     = contested_left,
        my_reachable_potential_eggs = my_reachable_potential_eggs,
        my_safe_potential_eggs      = my_safe_potential_eggs,
        sum_weighted_contested_dist = sum_weighted_contested_dist,
        bad_turd_count              = bad_turd_count,
        # Opponent fields
        opp_reachable_potential_eggs    = opp_reachable_potential_eggs,
        opp_safe_potential_eggs         = opp_safe_potential_eggs,
        opp_sum_weighted_contested_dist = opp_sum_weighted_contested_dist,
        opp_bad_turd_count              = opp_bad_turd_count,
        opp_min_egg_dist                = opp_min_egg_dist,
        owner                           = owner,
        region_sizes                    = region_stats,
    )




OWNER_NONE = 0
OWNER_ME   = 1
OWNER_OPP  = 2


class VoronoiInfo:
    """
    Board-dependent Voronoi / frontier analysis.

    Fields:
      - dist_me[x][y], dist_opp[x][y] : BFS distances for each player
      - owner[x][y]  : OWNER_NONE / OWNER_ME / OWNER_OPP (reachability-based)
      - my_owned     : # of cells where I arrive first (or tie)
      - opp_owned    : # of cells where opponent arrives first
      - contested    : # of "my frontier" squares:
                       both can reach, I own it, and opp is within 1 ply
      - max_contested_dist : max d_me over all contested squares
      - my_voronoi   : parity-based, corner-weighted voronoi score for me
      - opp_voronoi  : same for opponent
      - vor_score    : my_voronoi - opp_voronoi

      - contested_up    : how many contested squares lie primarily "above" me
      - contested_right : how many contested squares lie primarily "right" of me
      - contested_down  : how many contested squares lie primarily "below" me
      - contested_left  : how many contested squares lie primarily "left" of me

      - frag_directional: fragmentation score in [0,1] based on the distribution
                          of contested squares across directions.
    """

    __slots__ = (
        "my_owned",
        "opp_owned",
        "contested",
        "average_contested_dist",
        "max_contested_dist",
        "min_contested_dist",
        "min_egg_dist",
        "my_voronoi",
        "opp_voronoi",
        "vor_score",
        "contested_up",
        "contested_right",
        "contested_down",
        "contested_left",
        "my_reachable_potential_eggs",
        "my_safe_potential_eggs",
        "sum_weighted_contested_dist",
        "bad_turd_count",
        "opp_reachable_potential_eggs",
        "opp_safe_potential_eggs",
        "opp_sum_weighted_contested_dist",
        "opp_bad_turd_count",
        "opp_min_egg_dist",
        "owner",
        "region_sizes",
    )

    def __init__(
        self,
        my_owned,
        opp_owned,
        contested,
        average_contested_dist,
        max_contested_dist,
        min_contested_dist,
        min_egg_dist,
        my_voronoi,
        opp_voronoi,
        contested_up,
        contested_right,
        contested_down,
        contested_left,
        my_reachable_potential_eggs,
        my_safe_potential_eggs,
        sum_weighted_contested_dist,
        bad_turd_count,
        opp_reachable_potential_eggs,
        opp_safe_potential_eggs,
        opp_sum_weighted_contested_dist,
        opp_bad_turd_count,
        opp_min_egg_dist,
        owner,
        region_sizes,
    ):
        self.my_owned           = my_owned
        self.opp_owned          = opp_owned
        self.contested          = contested
        self.average_contested_dist = average_contested_dist
        self.max_contested_dist = max_contested_dist
        self.min_contested_dist = min_contested_dist
        self.min_egg_dist       = min_egg_dist
        self.my_voronoi  = my_voronoi
        self.opp_voronoi = opp_voronoi
        self.vor_score   = my_voronoi - opp_voronoi

        self.contested_up    = contested_up
        self.contested_right = contested_right
        self.contested_down  = contested_down
        self.contested_left  = contested_left

        self.my_reachable_potential_eggs = my_reachable_potential_eggs
        self.my_safe_potential_eggs = my_safe_potential_eggs
        self.sum_weighted_contested_dist = sum_weighted_contested_dist
        self.bad_turd_count = bad_turd_count
        
        self.opp_reachable_potential_eggs = opp_reachable_potential_eggs
        self.opp_safe_potential_eggs = opp_safe_potential_eggs
        self.opp_sum_weighted_contested_dist = opp_sum_weighted_contested_dist
        self.opp_bad_turd_count = opp_bad_turd_count
        self.opp_min_egg_dist = opp_min_egg_dist
        self.opp_bad_turd_count = opp_bad_turd_count
        self.opp_min_egg_dist = opp_min_egg_dist
        self.owner = owner
        self.region_sizes = region_sizes
