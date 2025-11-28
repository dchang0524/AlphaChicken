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
    # occupied = eggs + turds + traps
    # But we iterate x,y anyway.
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
    
    total_unclaimed_eggs = 0.0

    for x in range(dim):
        for y in range(dim):

            d_me  = dist_me[x][y]
            d_opp = dist_opp[x][y]

            reachable_me  = (d_me  >= 0)
            reachable_opp = (d_opp >= 0)
            
            # Calculate potential egg weight for this square
            is_corner = (x == 0 or x == last) and (y == 0 or y == last)
            w = 3 if is_corner else 1
            
            if is_empty(x, y):
                total_unclaimed_eggs += w

            # --------------------------
            # 1) Reachability ownership
            # --------------------------
            if reachable_me and (not reachable_opp or d_me <= d_opp):
                owner[x][y] = OWNER_ME

            elif reachable_opp:
                owner[x][y] = OWNER_OPP
            
            elif reachable_me and not reachable_opp:
                my_owned += 1
                
            # Count potential eggs (empty squares)
            # Must match my parity
            # Corner = 3, Edge/Center = 1
            if is_empty(x, y):
                # MY POTENTIAL EGGS
                if owner[x][y] == OWNER_ME:
                    # Check parity
                    min_egg_dist = min(min_egg_dist, d_me)
                    square_even = ((x + y) & 1) == 0
                    if square_even == my_even:
                        if reachable_opp:
                            my_reachable_potential_eggs += w
                        else:
                            my_safe_potential_eggs += w
                
                # OPP POTENTIAL EGGS
                elif owner[x][y] == OWNER_OPP:
                    # Check parity
                    opp_min_egg_dist = min(opp_min_egg_dist, d_opp)
                    square_even = ((x + y) & 1) == 0
                    if square_even == opp_even:
                        if reachable_me:
                            opp_reachable_potential_eggs += w
                        else:
                            opp_safe_potential_eggs += w

            # --------------------------
            # 2) Contested frontier
            #    - both can reach
            #    - I own it (I get there first or tie)
            #    - opp is within 1 ply of me
            # --------------------------
            if (
                reachable_me
                and reachable_opp
                and owner[x][y] == OWNER_ME
                and abs(d_me - d_opp) <= 1
            ):
                contested += 1
                average_contested_dist += d_me
                # Depth of my frontier from my POV
                if d_me > max_contested_dist:
                    max_contested_dist = d_me
                elif d_me < min_contested_dist:
                    min_contested_dist = d_me

                # Direction assignment from my position
                dx = x - my_x
                dy = y - my_y

                # Ignore my own square if it ever qualifies (paranoia guard)
                if dx != 0 or dy != 0:
                    #TODO: Check if its better to only add contested to the dominant direction
                    #if abs(dx) > abs(dy):
                    if dx > 0:
                        contested_right += 1
                    else:
                        contested_left += 1
                    #else:
                if dy > 0:
                    contested_down += 1   # y increasing = DOWN
                elif dy < 0:
                    contested_up += 1     # y decreasing = UP

            # --------------------------
            # 3) Parity-based Voronoi (corner-weighted)
            # --------------------------

            # Corner weight (already w)
            weight = w

            # square parity
            square_even = ((x + y) & 1) == 0
            is_my_par   = (square_even == my_even)
            is_opp_par  = (square_even == opp_even)

            # --- MY VORONOI ---
            if is_my_par and owner[x][y] == OWNER_ME:
                my_voronoi += weight

            # --- OPP VORONOI ---
            if is_opp_par and owner[x][y] == OWNER_OPP:
                opp_voronoi += weight

    average_contested_dist /= contested if contested > 0 else 64

    # --------------------------
    # 5) Region Size Heuristic
    # --------------------------
    weighted_contested = 0.0
    
    # Compute region sizes for all my owned cells
    region_sizes, region_ids = compute_region_sizes(cur_board, owner, OWNER_ME)

    # Sum up weighted contested score
    # We iterate over the board to find contested squares (which we identified earlier, but didn't store explicitly)
    # Alternatively, we can just iterate over the board again or store them in a list during the main loop.
    # Let's just iterate again, it's 8x8 or similar small size.
    
    if total_unclaimed_eggs == 0:
        total_unclaimed_eggs = 1.0

    # Store min distance for each contested region
    # region_id -> (min_dist, region_size)
    region_min_dists = {}

    for x in range(dim):
        for y in range(dim):
            # Check if it is a contested square
            # Condition: I own it, reachable by both, opp within 1 ply
            d_me  = dist_me[x][y]
            d_opp = dist_opp[x][y]
            
            if (
                d_me >= 0
                and d_opp >= 0
                and owner[x][y] == OWNER_ME
                and abs(d_me - d_opp) <= 1
            ):
                # It is contested.
                # Distance is d_me (distance from my chicken to this square)
                # RegionSize is region_sizes[x][y]
                
                r_size = region_sizes[x][y]
                rid    = region_ids[x][y]
                
                if r_size > 0 and rid > 0:
                    # Update min distance for this region
                    curr_min, _ = region_min_dists.get(rid, (float('inf'), 0))
                    if d_me < curr_min:
                        region_min_dists[rid] = (d_me, r_size)
                
                # Keep the old average metric as is (per square)
                if r_size > 0:
                     weighted_contested += r_size * (d_me)
                
                weighted_contested = weighted_contested / contested if contested > 0 else 0

    # Calculate sum based on unique regions
    for rid, (dist, r_size) in region_min_dists.items():
        term = (r_size / total_unclaimed_eggs) * dist
        sum_weighted_contested_dist += term
    sum_weighted_contested_dist = sum_weighted_contested_dist / contested if contested > 0 else 0
    
    # --- OPPONENT WEIGHTED CONTESTED ---
    # We need to do the same for opponent.
    # Opponent "contested" squares are ones THEY own, reachable by ME, and I am within 1 ply.
    # Wait, "contested" definition is symmetric?
    # "both can reach, I own it, and opp is within 1 ply".
    # For opp: "both can reach, OPP owns it, and I am within 1 ply".
    # Let's calculate opp regions first.
    opp_region_sizes, opp_region_ids = compute_region_sizes(cur_board, owner, OWNER_OPP)
    
    opp_region_min_dists = {}
    opp_contested_count = 0
    
    for x in range(dim):
        for y in range(dim):
            d_me  = dist_me[x][y]
            d_opp = dist_opp[x][y]
            
            if (
                d_me >= 0
                and d_opp >= 0
                and owner[x][y] == OWNER_OPP
                and abs(d_opp - d_me) <= 1
            ):
                opp_contested_count += 1
                r_size = opp_region_sizes[x][y]
                rid    = opp_region_ids[x][y]
                
                if r_size > 0 and rid > 0:
                    curr_min, _ = opp_region_min_dists.get(rid, (float('inf'), 0))
                    if d_opp < curr_min:
                        opp_region_min_dists[rid] = (d_opp, r_size)

    for rid, (dist, r_size) in opp_region_min_dists.items():
        term = (r_size / total_unclaimed_eggs) * dist
        opp_sum_weighted_contested_dist += term
    opp_sum_weighted_contested_dist = opp_sum_weighted_contested_dist / opp_contested_count if opp_contested_count > 0 else 0

    # --------------------------
    # 6) Bad Turd Calculation
    # --------------------------
    # A turd is "bad" (wasted) if all its neighbors are safe (unreachable by opponent).
    # We iterate over player's turds.
    for tx, ty in turds_me:
        is_useless = True
        for dx, dy in [(0, 2), (0, -2), (2, 0), (-2, 0), (1,1), (1, -1), (-1, 1), (-1, -1)]:
            nx, ny = tx + dx, ty + dy
            if 0 <= nx < dim and 0 <= ny < dim:
                # If neighbor is reachable by opponent (d_opp >= 0), then turd is blocking something useful.
                if dist_opp[nx][ny] >= 0:
                    is_useless = False
                    break
        
        if is_useless:
            bad_turd_count += 1
            
    # Opponent bad turds
    for tx, ty in turds_opp:
        is_useless = True
        for dx, dy in [(0, 2), (0, -2), (2, 0), (-2, 0), (1,1), (1, -1), (-1, 1), (-1, -1)]:
            nx, ny = tx + dx, ty + dy
            if 0 <= nx < dim and 0 <= ny < dim:
                # If neighbor is reachable by ME (d_me >= 0), then turd is blocking something useful.
                if dist_me[nx][ny] >= 0:
                    is_useless = False
                    break
        
        if is_useless:
            opp_bad_turd_count += 1

    return VoronoiInfo(
        owner              = owner,
        region_sizes       = region_sizes,
        my_owned           = my_owned,
        opp_owned          = opp_owned,
        contested          = contested,
        average_contested_dist = average_contested_dist,
        max_contested_dist = max_contested_dist,
        min_contested_dist = min_contested_dist,
        min_egg_dist       = min_egg_dist,
        my_voronoi=  my_voronoi,
        opp_voronoi = opp_voronoi,
        contested_up       = contested_up,
        contested_right    = contested_right,
        contested_down     = contested_down,
        contested_left     = contested_left,
        my_reachable_potential_eggs = my_reachable_potential_eggs,
        my_safe_potential_eggs = my_safe_potential_eggs,
        sum_weighted_contested_dist = sum_weighted_contested_dist,
        bad_turd_count = bad_turd_count,
        # Opponent fields
        opp_reachable_potential_eggs = opp_reachable_potential_eggs,
        opp_safe_potential_eggs = opp_safe_potential_eggs,
        opp_sum_weighted_contested_dist = opp_sum_weighted_contested_dist,
        opp_bad_turd_count = opp_bad_turd_count,
        opp_min_egg_dist = opp_min_egg_dist,
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
        "owner",
        "region_sizes",
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
    )

    def __init__(
        self,
        owner,
        region_sizes,
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
    ):
        self.owner              = owner
        self.region_sizes       = region_sizes
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

