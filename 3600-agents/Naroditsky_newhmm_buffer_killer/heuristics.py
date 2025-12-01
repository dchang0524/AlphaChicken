import math
from typing import List, Tuple, Optional

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
    killer_move: Optional[Move] = None  # <--- NEW ARGUMENT
) -> List[Move]:
    """
    Order moves with Killer Heuristic priority.
    """

    if not moves:
        return moves

    dim = cur_board.game_map.MAP_SIZE

    # 1) Basic filtering (Drop TURD if no contested, Drop PLAIN if EGG exists)
    filtered = list(moves)
    total_contested = getattr(vor, "contested", 0)

    if total_contested == 0:
        filtered = [mv for mv in filtered if mv[1] != MoveType.TURD]

    has_egg = any(mt == MoveType.EGG for _, mt in filtered)
    if has_egg:
        filtered = [mv for mv in filtered if mv[1] != MoveType.PLAIN]

    if not filtered:
        filtered = list(moves)

    # ----------------------------
    # 2) KILLER MOVE LOGIC (NEW)
    # ----------------------------
    killers = []
    
    # If the killer move is in our valid list, prioritize it immediately
    if killer_move and killer_move in filtered:
        killers.append(killer_move)
        filtered.remove(killer_move)

    # ----------------------------
    # 3) Standard Scoring
    # ----------------------------
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

    BASE_EGG   = 3.0
    BASE_PLAIN = 2.0
    BASE_TURD  = 1.0
    DIR_WEIGHT = 0.5
    TURD_CENTER_BONUS   = 2.0
    TURD_FRONTIER_BONUS = 3.0

    scored: List[tuple[float, Move]] = []

    for mv in filtered:
        direction, mtype = mv

        if mtype == MoveType.EGG:
            score = BASE_EGG
        elif mtype == MoveType.PLAIN:
            score = BASE_PLAIN
        else:
            score = BASE_TURD

        dir_contested = contested_by_dir.get(direction, 0)
        score += DIR_WEIGHT * dir_contested

        if mtype == MoveType.TURD:
            if my_center_dist <= 2:
                score += TURD_CENTER_BONUS
            if near_frontier:
                score += TURD_FRONTIER_BONUS

        scored.append((score, mv))

    scored.sort(key=lambda x: x[0], reverse=True)
    
    # Return Killers first, then the rest sorted
    return killers + [mv for _, mv in scored]