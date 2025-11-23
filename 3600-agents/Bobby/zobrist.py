# zobrist.py

from __future__ import annotations

import random
from typing import List, Tuple, Set
from game import board as board_mod  # type: ignore

# Feature indices (POV-relative)
Z_ME_EGG      = 0
Z_OPP_EGG     = 1
Z_ME_TURD     = 2
Z_OPP_TURD    = 3
Z_ME_CHICKEN  = 4
Z_OPP_CHICKEN = 5
Z_KNOWN_TRAP  = 6

Z_NUM_FEATURES = 7

# Global tables
_ZOBRIST_TABLE: List[List[List[int]]] | None = None
_ZOBRIST_SIDE_TO_MOVE: int | None = None


def init_zobrist(dim: int = 8, seed: int = 1234567) -> None:
    """
    Initialize Zobrist random bitstrings.

    Must be called once at startup (before using zobrist_hash).
    Using a fixed seed makes behavior deterministic for debugging.
    """
    global _ZOBRIST_TABLE, _ZOBRIST_SIDE_TO_MOVE

    rnd = random.Random(seed)
    _ZOBRIST_TABLE = [
        [
            [rnd.getrandbits(64) for _ in range(Z_NUM_FEATURES)]
            for _ in range(dim)
        ]
        for _ in range(dim)
    ]
    _ZOBRIST_SIDE_TO_MOVE = rnd.getrandbits(64)


def zobrist_hash(
    cur_board: board_mod.Board,
    known_traps: Set[Tuple[int, int]],
) -> int:
    """
    Compute Zobrist hash for the CURRENT POV board + known_traps.

    - POV: board.chicken_player is "me", board.chicken_enemy is "opp".
    - eggs_player / turds_player belong to "me".
    - eggs_enemy / turds_enemy belong to "opp".
    - known_traps is the agent's collapsed belief about discovered trapdoors.
      (If you don't want TT to depend on belief, pass an empty set.)

    This recomputes from scratch; incremental updates are optional later.
    """
    global _ZOBRIST_TABLE, _ZOBRIST_SIDE_TO_MOVE

    if _ZOBRIST_TABLE is None or _ZOBRIST_SIDE_TO_MOVE is None:
        raise RuntimeError("Zobrist not initialized. Call init_zobrist() first.")

    table = _ZOBRIST_TABLE
    dim = cur_board.game_map.MAP_SIZE

    h = 0

    eggs_me   = cur_board.eggs_player
    eggs_opp  = cur_board.eggs_enemy
    turds_me  = cur_board.turds_player
    turds_opp = cur_board.turds_enemy

    me_pos  = cur_board.chicken_player.get_location()
    opp_pos = cur_board.chicken_enemy.get_location()

    # --- Board pieces ---

    for x in range(dim):
        for y in range(dim):
            pos = (x, y)

            cell_key = 0

            if pos == me_pos:
                cell_key ^= table[x][y][Z_ME_CHICKEN]

            if pos == opp_pos:
                cell_key ^= table[x][y][Z_OPP_CHICKEN]

            if pos in eggs_me:
                cell_key ^= table[x][y][Z_ME_EGG]

            if pos in eggs_opp:
                cell_key ^= table[x][y][Z_OPP_EGG]

            if pos in turds_me:
                cell_key ^= table[x][y][Z_ME_TURD]

            if pos in turds_opp:
                cell_key ^= table[x][y][Z_OPP_TURD]

            if pos in known_traps:
                cell_key ^= table[x][y][Z_KNOWN_TRAP]

            h ^= cell_key

    # Side to move: for POV-board, current player is always "me".
    # If you ever store non-POV boards, XOR this to distinguish.
    h ^= _ZOBRIST_SIDE_TO_MOVE

    return h

class TTEntry:
    __slots__ = ("value", "depth", "flag", "best_move")

    def __init__(self, value: float, depth: int, flag: str, best_move):
        self.value = value
        self.depth = depth
        self.flag = flag     # "EXACT" / "LOWER" / "UPPER"
        self.best_move = best_move
