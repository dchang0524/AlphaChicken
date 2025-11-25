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
    Optimized Sparse Zobrist Hash.
    Iterates over piece sets instead of the 8x8 grid.
    Speedup: ~20x faster than the dense loop.
    """
    global _ZOBRIST_TABLE, _ZOBRIST_SIDE_TO_MOVE

    if _ZOBRIST_TABLE is None or _ZOBRIST_SIDE_TO_MOVE is None:
        raise RuntimeError("Zobrist not initialized. Call init_zobrist() first.")

    table = _ZOBRIST_TABLE
    h = 0

    # 1. Chickens (Always 2)
    mx, my = cur_board.chicken_player.get_location()
    h ^= table[mx][my][Z_ME_CHICKEN]

    ox, oy = cur_board.chicken_enemy.get_location()
    h ^= table[ox][oy][Z_OPP_CHICKEN]

    # 2. My Eggs (Sparse Set Iteration)
    for x, y in cur_board.eggs_player:
        h ^= table[x][y][Z_ME_EGG]

    # 3. Opp Eggs
    for x, y in cur_board.eggs_enemy:
        h ^= table[x][y][Z_OPP_EGG]

    # 4. My Turds
    for x, y in cur_board.turds_player:
        h ^= table[x][y][Z_ME_TURD]

    # 5. Opp Turds
    for x, y in cur_board.turds_enemy:
        h ^= table[x][y][Z_OPP_TURD]

    # 6. Known Traps
    for x, y in known_traps:
        h ^= table[x][y][Z_KNOWN_TRAP]

    # 7. Side to move (Constant XOR)
    h ^= _ZOBRIST_SIDE_TO_MOVE

    return h

class TTEntry:
    __slots__ = ("value", "depth", "flag", "best_move", "gen")

    def __init__(self, value: float, depth: int, flag: str, best_move, gen: int):
        self.value = value
        self.depth = depth
        self.flag = flag     # "EXACT" / "LOWER" / "UPPER"
        self.best_move = best_move
        self.gen = gen