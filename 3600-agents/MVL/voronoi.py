from __future__ import annotations

from typing import List, Tuple, Set
from game import board as board_mod  # type: ignore

from .bfs_utils import bfs_distances_both  # assumes you defined this in bfs_utils.py

OWNER_NONE = 0
OWNER_ME   = 1  # current player (board.chicken_player)
OWNER_OPP  = 2  # enemy (board.chicken_enemy)

# NEW: Threshold to detect walls/unreachable squares from the Lava BFS
MAX_REACHABLE = 10000.0

def analyze(
    self,
    cur_board: board_mod.Board,
    known_traps: Set[Tuple[int, int]],
) -> VoronoiInfo:

    dim  = cur_board.game_map.MAP_SIZE
    last = dim - 1

    # 1) BFS distances 
    # FIX 1: Pass self.trap_belief to the new BFS signature
    dist_me, dist_opp = bfs_distances_both(cur_board, known_traps, self.trap_belief)

    # Parities
    my_even  = cur_board.chicken_player.even_chicken
    opp_even = cur_board.chicken_enemy.even_chicken

    # My current position (for directional contested counts)
    my_x, my_y = cur_board.chicken_player.get_location()

    # 2D owner grid
    owner: List[List[int]] = [
        [OWNER_NONE for _ in range(dim)] for _ in range(dim)
    ]

    my_owned   = 0
    opp_owned  = 0
    contested  = 0
    
    # FIX 2: Initialize min/max with floats and large values suitable for weighted graphs
    max_contested_dist = 0.0
    min_contested_dist = 99999.0
    min_egg_dist       = 99999.0

    my_voronoi  = 0
    opp_voronoi = 0

    # Directional frontier pressure
    contested_up = 0
    contested_right = 0
    contested_down = 0
    contested_left = 0

    #Quadrant Fragmentation Score
    contested_q1 = 0
    contested_q2 = 0
    contested_q3 = 0
    contested_q4 = 0

    for x in range(dim):
        for y in range(dim):

            d_me  = dist_me[x][y]
            d_opp = dist_opp[x][y]

            # FIX 3: Check against MAX_REACHABLE, not just >= 0
            # Walls are now 999999.0, so >= 0 would count walls as reachable!
            reachable_me  = (d_me  < MAX_REACHABLE)
            reachable_opp = (d_opp < MAX_REACHABLE)

            # --------------------------
            # 1) Reachability ownership
            # --------------------------
            if reachable_me and (not reachable_opp or d_me <= d_opp):
                owner[x][y] = OWNER_ME
                my_owned += 1

            elif reachable_opp:
                owner[x][y] = OWNER_OPP
                opp_owned += 1

            # --------------------------
            # 2) Contested frontier
            #    - both can reach
            #    - I own it (I get there first or tie)
            #    - opp is within 1 ply (approx 1.5 cost) of me
            # --------------------------
            if (
                reachable_me
                and reachable_opp
                and owner[x][y] == OWNER_ME
                # Float tolerance: <= 1.5 ensures we catch "1 step" even with precision noise
                and abs(d_me - d_opp) <= 1.5
            ):
                contested += 1

                # Depth of my frontier from my POV
                if d_me > max_contested_dist:
                    max_contested_dist = d_me
                if d_me < min_contested_dist:
                    min_contested_dist = d_me

                # Direction assignment from my position
                dx = x - my_x
                dy = y - my_y

                # Ignore my own square if it ever qualifies (paranoia guard)
                if dx != 0 or dy != 0:
                    if dx > 0:
                        contested_right += 1
                    else:
                        contested_left += 1
                if dy > 0:
                    contested_down += 1   # y increasing = DOWN
                elif dy < 0:
                    contested_up += 1     # y decreasing = UP
                
                # Quadrant
                if dx > 0 and dy < 0:
                    contested_q1 += 1
                elif dx < 0 and dy < 0:
                    contested_q2 += 1
                elif dx < 0 and dy > 0:
                    contested_q3 += 1
                elif dx > 0 and dy > 0:
                    contested_q4 += 1

            # --------------------------
            # 3) Parity-based Voronoi (corner-weighted)
            # --------------------------

            # Corner weight
            is_corner = (x == 0 or x == last) and (y == 0 or y == last)
            weight = 3 if is_corner else 1

            # square parity
            square_even = ((x + y) & 1) == 0
            is_my_par   = (square_even == my_even)
            is_opp_par  = (square_even == opp_even)

            # --- MY VORONOI ---
            if is_my_par and owner[x][y] == OWNER_ME:
                my_voronoi += weight
                min_egg_dist = min(min_egg_dist, dist_me[x][y])

            # --- OPP VORONOI ---
            if is_opp_par and owner[x][y] == OWNER_OPP:
                opp_voronoi += weight

    # 4) Directional fragmentation score in [0,1], based on contested_* counts
    L = contested_left
    R = contested_right
    U = contested_up
    D = contested_down

    counts = [L, R, U, D]
    total = sum(counts)

    if total <= 1:
        cardinal_frag = 0.0
    else:
        # How many directions actually have contested squares?
        dir_count = sum(1 for c in counts if c > 0)
        # Normalize: 1 direction → 0, 4 directions → 1
        dir_score = (dir_count - 1) / 3.0  # ∈ [0,1]

        # How evenly is the frontier spread?
        major_fraction = max(counts) / total  # ∈ (0,1]
        spread_score = 1.0 - major_fraction   # ∈ [0,1), 0 when all in one dir

        # Opposite-direction bonus: L+R or U+D both active → more fragmentation.
        opp_bonus = 0.0
        if L > 0 and R > 0:
            opp_bonus += 0.5
        if U > 0 and D > 0:
            opp_bonus += 0.5
        opp_score = min(1.0, opp_bonus)  # 0, 0.5, or 1.0

        cardinal_frag = (
            0.4 * spread_score +   # how evenly spread
            0.2 * dir_score   +    # how many directions
            0.4 * opp_score        # opposite sides involved
        )
        # Clamp
        if cardinal_frag < 0.0:
            cardinal_frag = 0.0
        elif cardinal_frag > 1.0:
            cardinal_frag = 1.0

    quad_counts = [contested_q1, contested_q2, contested_q3, contested_q4]
    quad_dirs = sum(1 for c in quad_counts if c > 0)
    quad_spread = 1.0 - (max(quad_counts) / total) if total > 0 else 0.0
    quad_score = 0.5 * quad_spread + 0.5 * ((quad_dirs - 1) / 3.0)
    frag_score = 0.5 * cardinal_frag + 0.5 * quad_score

    return VoronoiInfo(
        my_owned           = my_owned,
        opp_owned          = opp_owned,
        contested          = contested,
        max_contested_dist = max_contested_dist,
        min_contested_dist = min_contested_dist,
        min_egg_dist       = min_egg_dist,
        my_voronoi  = my_voronoi,
        opp_voronoi = opp_voronoi,
        contested_up       = contested_up,
        contested_right    = contested_right,
        contested_down     = contested_down,
        contested_left     = contested_left,
        frag_score = frag_score,
    )


OWNER_NONE = 0
OWNER_ME   = 1
OWNER_OPP  = 2


class VoronoiInfo:
    """
    Board-dependent Voronoi / frontier analysis.
    Stores ONLY scalar metrics. NO GRIDS allowed here to prevent Memory Error.
    """

    __slots__ = (
        "owner",
        "my_owned",
        "opp_owned",
        "contested",
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
        "frag_score",
    )

    def __init__(
        self,
        my_owned,
        opp_owned,
        contested,
        max_contested_dist,
        min_contested_dist,
        min_egg_dist,
        my_voronoi,
        opp_voronoi,
        contested_up,
        contested_right,
        contested_down,
        contested_left,
        frag_score,
    ):
        self.my_owned           = my_owned
        self.opp_owned          = opp_owned
        self.contested          = contested
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

        self.frag_score = frag_score