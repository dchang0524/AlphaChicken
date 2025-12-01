import math
from typing import List, Tuple

from game import board as board_mod  # type: ignore
from .voronoi import VoronoiInfo, OWNER_ME, OWNER_OPP  # adjust import path as needed
from .hiddenMarkov import TrapdoorBelief
from game.enums import Direction, MoveType  # type: ignore

INF = 10 ** 8
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

