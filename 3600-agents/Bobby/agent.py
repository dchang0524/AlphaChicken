from __future__ import annotations

from typing import List, Tuple, Dict, Optional, Set
from collections.abc import Callable
import math

# Game engine imports
from game import board as board_mod  # type: ignore
from game.enums import MoveType, Direction  # type: ignore
from game.game_map import GameMap  # optional; remove if unused

# Your modules
from .voronoi import analyze as voronoi_analyze, VoronoiInfo
from .hiddenMarkov import TrapdoorBelief
from .zobrist import init_zobrist, zobrist_hash, TTEntry
from .heuristics import evaluate, move_order

# Utilities
import numpy as np

INF = 10 ** 8

INF = 10 ** 8

class PlayerAgent:
    def __init__(self, initial_board: board_mod.Board, time_left: Callable):
        dim = initial_board.game_map.MAP_SIZE

        self.trap_belief = TrapdoorBelief(dim)

        # Deterministic trapdoors (walls for BFS / planning / zobrist)
        self.known_traps: set[Tuple[int, int]] = set()

        # Probabilistic candidates for expectimax
        self.potential_even: list[Tuple[int, int]] = []
        self.potential_odd:  list[Tuple[int, int]] = []

        self.max_depth = 10

        # Transposition table: search metadata only
        self.tt: Dict[int, TTEntry] = {}

        # Optional Voronoi cache to avoid re-running BFS for same state
        self.vor_cache: Dict[int, VoronoiInfo] = {}

        self.last_root_best: Optional[Tuple] = None

        init_zobrist(dim, seed=1234567)

    # ------------------------------------------------------------------
    # Main entry: play
    # ------------------------------------------------------------------
    def play(
        self,
        board: board_mod.Board,
        sensor_data: List[Tuple[bool, bool]],
        time_left: Callable,
    ):
        # 1) Update deterministic trapdoors from engine
        engine_found = set(getattr(board, "found_trapdoors", set()))
        new_found = engine_found - self.known_traps
        for pos in new_found:
            self.trap_belief.set_trapdoor(pos)
        self.known_traps |= engine_found

        # 2) HMM update from sensors
        my_pos  = board.chicken_player.get_location()
        opp_pos = board.chicken_enemy.get_location()

        self.trap_belief.update(my_pos, sensor_data)
        self.trap_belief.mark_safe(my_pos)
        self.trap_belief.mark_safe(opp_pos)

        # 3) Extract probabilistic trap candidates for expectimax (do NOT touch known_traps)
        self._update_potential_traps(board, threshold=0.30)

        # 4) Clear TT & Voronoi cache for this root search
        self.tt.clear()
        self.vor_cache.clear()

        # 5) Legal moves
        moves = board.get_valid_moves()
        if not moves:
            return None

        # 6) Iterative deepening with expectimax over (even_trap, odd_trap)
        target_depth = self._choose_max_depth(board)
        best_move = None
        best_val = -INF

        for depth in range(1, target_depth + 1):
            if time_left() < 0.05:
                break

            val, mv = self._search_root(board, moves, depth, time_left)
            if mv is None:
                break

            best_val = val
            best_move = mv
            self.last_root_best = mv

        if best_move is None:
            best_move = moves[0]

        return best_move

    # ------------------------------------------------------------------
    # Potential trap candidates (for expectimax)
    # ------------------------------------------------------------------
    def _update_potential_traps(
        self,
        board: board_mod.Board,
        threshold: float = 0.30,
    ) -> None:
        """
        Collect candidate trap squares (even and odd parity) with belief >= threshold.
        These are used for expectimax over trap placements, but remain traversable
        for BFS and are NOT added to known_traps.
        """
        dim = board.game_map.MAP_SIZE
        self.potential_even = []
        self.potential_odd  = []

        for x in range(dim):
            for y in range(dim):
                p = self.trap_belief.prob_at((x, y))
                if p < threshold:
                    continue
                if ((x + y) & 1) == 0:
                    self.potential_even.append((x, y))
                else:
                    self.potential_odd.append((x, y))

    # ------------------------------------------------------------------
    # Voronoi cache
    # ------------------------------------------------------------------
    def _get_voronoi(self, cur_board: board_mod.Board) -> VoronoiInfo:
        """
        Fetch VoronoiInfo from cache or compute it.
        Keyed by board+known_traps via Zobrist.
        """
        key = zobrist_hash(cur_board, self.known_traps)
        vor = self.vor_cache.get(key)
        if vor is None:
            vor = voronoi_analyze(self, cur_board, self.known_traps)
            self.vor_cache[key] = vor
        return vor

    # ------------------------------------------------------------------
    # Root search with expectimax over trap configs
    # ------------------------------------------------------------------
    def _build_trap_scenarios(
        self,
    ) -> List[Tuple[Optional[Tuple[int, int]], Optional[Tuple[int, int]], float]]:
        """
        Build (even_trap, odd_trap, weight) scenarios from potential_even/odd.

        - If no candidates for a parity, we use [None] with prob 1.0 for that parity.
        - We normalize weights so they sum to 1.
        """
        evens = list(self.potential_even)
        odds  = list(self.potential_odd)

        # If no info at all, single 'no trap info' scenario.
        if not evens and not odds:
            return [(None, None, 1.0)]

        # Even side
        if not evens:
            evens = [None]
            even_probs = [1.0]
        else:
            even_probs = [self.trap_belief.prob_at(pos) for pos in evens]
            s = sum(even_probs)
            if s <= 0.0:
                even_probs = [1.0 / len(evens)] * len(evens)
            else:
                even_probs = [p / s for p in even_probs]

        # Odd side
        if not odds:
            odds = [None]
            odd_probs = [1.0]
        else:
            odd_probs = [self.trap_belief.prob_at(pos) for pos in odds]
            s = sum(odd_probs)
            if s <= 0.0:
                odd_probs = [1.0 / len(odds)] * len(odds)
            else:
                odd_probs = [p / s for p in odd_probs]

        scenarios: List[Tuple[Optional[Tuple[int, int]], Optional[Tuple[int, int]], float]] = []
        for e, pe in zip(evens, even_probs):
            for o, po in zip(odds, odd_probs):
                w = pe * po
                scenarios.append((e, o, w))

        total_w = sum(w for _, _, w in scenarios)
        if total_w > 0.0:
            scenarios = [(e, o, w / total_w) for (e, o, w) in scenarios]
        else:
            # Degenerate; just flatten to one dummy scenario
            scenarios = [(None, None, 1.0)]

        return scenarios

    def _search_root(
        self,
        board: board_mod.Board,
        moves: List[Tuple],
        depth: int,
        time_left: Callable,
    ) -> Tuple[float, Optional[Tuple]]:
        """
        One iteration of iterative deepening with expectimax over trap configs.

        For each move:
          value(move) = sum_s P(s) * Negamax(board_after_move, s)
        """
        alpha, beta = -INF, INF
        best_move: Optional[Tuple] = None
        best_val = -INF

        # Voronoi once at root for move ordering
        vor_root = self._get_voronoi(board)
        ordered_moves = move_order(board, moves, vor_root)

        scenarios = self._build_trap_scenarios()

        for mv in ordered_moves:
            if time_left() < 0.02:
                break

            exp_val = 0.0

            for (even_trap, odd_trap, weight) in scenarios:
                # simulate move under this trap configuration
                child = self._simulate_move_with_traps(board, mv, even_trap, odd_trap)
                val = -self.negamax(child, depth - 1, -beta, -alpha,
                                    time_left, even_trap, odd_trap)
                exp_val += weight * val

            if exp_val > best_val:
                best_val = exp_val
                best_move = mv

            if best_val > alpha:
                alpha = best_val
            if alpha >= beta:
                break

        if best_move is None:
            return -INF, None
        return best_val, best_move

    # ------------------------------------------------------------------
    # Move simulation with trap effect
    # ------------------------------------------------------------------
    def _simulate_move_with_traps(
        self,
        cur_board: board_mod.Board,
        mv: Tuple,
        even_trap: Optional[Tuple[int, int]],
        odd_trap: Optional[Tuple[int, int]],
    ) -> board_mod.Board:
        """
        Forecast a move and apply teleporter effects for this scenario,
        then reverse perspective so the resulting board is from the POV
        of the next player to move.

        Correct teleporter rule:
            - If the *current moving player* lands on *either* trap (even_trap or odd_trap),
            they get teleported and the opponent gains eggs.
            - Trap parity does NOT matter. Any player stepping on any trap triggers it.
        """
        # mv = (direction, move_type)
        direction, move_type = mv

        # Forecast move correctly
        child = cur_board.forecast_move(direction, move_type)

        px, py = child.chicken_player.get_location()

        # Check if player stepped on *any* hypothetical trap this scenario assigns.
        stepped_on_even = (even_trap is not None and (px, py) == even_trap)
        stepped_on_odd  = (odd_trap  is not None and (px, py) == odd_trap)

        if stepped_on_even or stepped_on_odd:
            # Opponent gains eggs
            child.chicken_enemy.increment_eggs_laid(4)

            # Player gets teleported back to spawn
            child.chicken_player.reset_location()

        # Switch perspective for negamax child
        child.reverse_perspective()

        return child


    # ------------------------------------------------------------------
    # Negamax with TT (fixed trap scenario)
    # ------------------------------------------------------------------
    def negamax(
        self,
        cur_board: board_mod.Board,
        depth: int,
        alpha: float,
        beta: float,
        time_left: Callable,
        even_trap: Optional[Tuple[int, int]],
        odd_trap: Optional[Tuple[int, int]],
    ) -> float:
        """
        Negamax with side-to-move evaluation.
        Trap configuration (even_trap, odd_trap) is fixed along this subtree.
        """

        # Terminal: engine game over
        if cur_board.is_game_over():
            my_eggs  = cur_board.chicken_player.eggs_laid
            opp_eggs = cur_board.chicken_enemy.eggs_laid
            if my_eggs > opp_eggs:
                return INF
            elif my_eggs < opp_eggs:
                return -INF
            else:
                return 0.0

        # Emergency time cutoff
        if time_left() < 0.01:
            vor = self._get_voronoi(cur_board)
            return evaluate(cur_board, vor, self.trap_belief)

        alpha_orig = alpha

        # TT key must depend on trap configuration AND board
        base_key = zobrist_hash(cur_board, self.known_traps)
        scenario_key = hash((even_trap, odd_trap))
        key = base_key ^ scenario_key

        entry = self.tt.get(key)
        if entry is not None and entry.depth >= depth:
            if entry.flag == "EXACT":
                return entry.value
            elif entry.flag == "LOWER" and entry.value > alpha:
                alpha = entry.value
            elif entry.flag == "UPPER" and entry.value < beta:
                beta = entry.value
            if alpha >= beta:
                return entry.value

        vor = self._get_voronoi(cur_board)

        moves = cur_board.get_valid_moves()
        if depth == 0 or not moves:
            return evaluate(cur_board, vor, self.trap_belief)

        ordered_moves = self._order_moves(cur_board, moves, vor)

        best_val = -INF
        best_move = None

        for mv in ordered_moves:
            if time_left() < 0.01:
                break

            child = self._simulate_move_with_traps(cur_board, mv, even_trap, odd_trap)

            val = -self.negamax(child, depth - 1, -beta, -alpha,
                                time_left, even_trap, odd_trap)

            if val > best_val:
                best_val = val
                best_move = mv

            if best_val > alpha:
                alpha = best_val
            if alpha >= beta:
                break

        if best_move is not None:
            if best_val <= alpha_orig:
                flag = "UPPER"
            elif best_val >= beta:
                flag = "LOWER"
            else:
                flag = "EXACT"
            self.tt[key] = TTEntry(best_val, depth, flag, best_move)

        return best_val

    # ------------------------------------------------------------------
    # Internal move ordering wrapper (non-root)
    # ------------------------------------------------------------------
    def _order_moves(
        self,
        cur_board: board_mod.Board,
        moves: List[Tuple],
        vor: VoronoiInfo,
    ) -> List[Tuple]:
        return move_order(cur_board, moves, vor)

    # ------------------------------------------------------------------
    # Depth scheduling
    # ------------------------------------------------------------------
    def _choose_max_depth(self, board: board_mod.Board) -> int:
        """
        Choose search depth based on remaining time and whether the
        position is open or closed (no contested squares → deeper).
        """
        t = board.player_time  # seconds left, presumably

        # Base depth from clock
        if t < 5:
            base = 4
        elif t < 15:
            base = self.max_depth - 3
        elif t < 40:
            base = self.max_depth - 2
        elif t < 180:
            base = self.max_depth - 1
        else:
            base = self.max_depth

        # If position is closed (no contested), searching deeper is cheaper
        vor = self._get_voronoi(board)
        if vor.contested == 0:
            base = min(self.max_depth, base + 2)

        return min(base, self.max_depth)

