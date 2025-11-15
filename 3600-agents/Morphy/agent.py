from collections.abc import Callable
from typing import List, Tuple

import math
import numpy as np
from game import board  # type: ignore
from game.enums import MoveType, Direction  # type: ignore


class TrapdoorBelief:
    """
    Naive Bayesian belief over the two trapdoors.

    We maintain two distributions:
      - p_a over all EVEN squares (trapdoor A)
      - p_b over all ODD squares (trapdoor B)

    Each turn we update using (heard, felt) and distance from our current position.
    """

    def __init__(self, map_size: int):
        self.map_size = map_size
        self.even_squares = []
        self.odd_squares = []
        for x in range(map_size):
            for y in range(map_size):
                if (x + y) % 2 == 0:
                    self.even_squares.append((x, y))
                else:
                    self.odd_squares.append((x, y))

        self.p_a = {pos: 1.0 / len(self.even_squares) for pos in self.even_squares}
        self.p_b = {pos: 1.0 / len(self.odd_squares) for pos in self.odd_squares}

    @staticmethod
    def manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    @staticmethod
    def prob_hear(dist: int) -> float:
        """
        Approximate P(hear | distance). Replace with exact table from spec if you want.
        """
        if dist == 0:
            return 0.6
        if dist == 1:
            return 0.5
        if dist == 2:
            return 0.25
        if dist == 3:
            return 0.1
        return 0.0

    @staticmethod
    def prob_feel(dist: int) -> float:
        """
        Approximate P(feel | distance). More local than hearing.
        """
        if dist == 0:
            return 0.35
        if dist == 1:
            return 0.2
        if dist == 2:
            return 0.05
        return 0.0

    def _update_map(
        self,
        belief: dict[Tuple[int, int], float],
        my_pos: Tuple[int, int],
        heard: bool,
        felt: bool,
    ) -> dict[Tuple[int, int], float]:
        new_belief: dict[Tuple[int, int], float] = {}
        for pos, prior in belief.items():
            d = self.manhattan(my_pos, pos)
            ph = self.prob_hear(d)
            pf = self.prob_feel(d)

            lh = ph if heard else (1.0 - ph)
            lf = pf if felt else (1.0 - pf)
            likelihood = lh * lf

            new_belief[pos] = prior * likelihood

        total = sum(new_belief.values())
        if total <= 0.0:
            # Numerical safety: revert to uniform on that parity
            n = len(new_belief)
            if n == 0:
                return new_belief
            uniform = 1.0 / n
            return {pos: uniform for pos in new_belief}

        for pos in new_belief:
            new_belief[pos] /= total
        return new_belief

    def update(self, my_pos: Tuple[int, int], sensor_data: List[Tuple[bool, bool]]) -> None:
        """
        sensor_data[0] = (heard, felt) for trapdoor A (even square)
        sensor_data[1] = (heard, felt) for trapdoor B (odd square)
        """
        (heard_a, felt_a), (heard_b, felt_b) = sensor_data
        self.p_a = self._update_map(self.p_a, my_pos, heard_a, felt_a)
        self.p_b = self._update_map(self.p_b, my_pos, heard_b, felt_b)

    def prob_at(self, pos: Tuple[int, int]) -> float:
        """
        Total probability that `pos` is any trapdoor.
        """
        if (pos[0] + pos[1]) % 2 == 0:
            return self.p_a.get(pos, 0.0)
        else:
            return self.p_b.get(pos, 0.0)


class PlayerAgent:
    """
    __init__ and play signatures must match the harness.
    """

    # Heuristic weights – tune these
    # Heuristic weights
    EGGS_WEIGHT = 100.0        # giant weight for actual eggs
    POTENTIAL_WEIGHT = 2.0     # modest weight for future eggs
    MOBILITY_WEIGHT = 0.3      # tiny, just to avoid zugzwang feel
    CENTER_WEIGHT = 0.1        # really minor

    # Trapdoors = disaster
    TRAPDOOR_WEIGHT = 200.0
    TRAPDOOR_HARD_THRESHOLD = 0.8

    FIXED_TARGET_EGGS = 25
    BASE_DEPTH = 5
    LOW_TIME_DEPTH = 1
    CRITICAL_TIME_SECONDS = 3.0

    def __init__(self, board: board.Board, time_left: Callable):
        self.map_size = board.game_map.MAP_SIZE
        self.center = ((self.map_size - 1) / 2.0, (self.map_size - 1) / 2.0)

        # Precompute centrality weight grid
        self.center_weight_grid: dict[Tuple[int, int], float] = {}
        for x in range(self.map_size):
            for y in range(self.map_size):
                self.center_weight_grid[(x, y)] = -self.manhattan_dist((x, y), self.center)

        # Global trapdoor belief carried across turns
        self.trap_belief = TrapdoorBelief(self.map_size)

    # ================= PUBLIC ENTRY ================= #

    def play(
        self,
        board: board.Board,
        sensor_data: List[Tuple[bool, bool]],
        time_left: Callable,
    ):
        # 1) Update trapdoor beliefs
        my_pos = board.chicken_player.get_location()
        self.trap_belief.update(my_pos, sensor_data)

        # 2) Get legal moves
        moves = board.get_valid_moves()
        if not moves:
            return None
        if len(moves) == 1:
            return moves[0]

        tl = time_left()
        depth = self.choose_search_depth(tl)

        # When almost out of time, no deep search
        if tl < self.CRITICAL_TIME_SECONDS:
            return self.choose_greedy(board, moves)

        # 3) Negamax + alpha–beta
        best_move = None
        alpha = -math.inf
        beta = math.inf

        ordered_moves = self.order_moves(board, moves)

        for move in ordered_moves:
            if time_left() < 0.1:
                break
            child = self.simulate_move(board, move)
            score = -self.negamax(child, depth - 1, -beta, -alpha, time_left)
            if best_move is None or score > alpha:
                alpha = score
                best_move = move
            if alpha >= beta:
                break

        if best_move is None:
            best_move = moves[np.random.randint(len(moves))]

        print(f"I'm at {my_pos}.")
        print(f"Trapdoor A: heard? {sensor_data[0][0]}, felt? {sensor_data[0][1]}")
        print(f"Trapdoor B: heard? {sensor_data[1][0]}, felt? {sensor_data[1][1]}")
        return best_move

    # ================= SEARCH ================= #

    def choose_search_depth(self, time_left_val: float) -> int:
        depth = self.BASE_DEPTH
        if time_left_val < 60:
            depth = max(depth - 1, self.LOW_TIME_DEPTH)
        if time_left_val < 15:
            depth = self.LOW_TIME_DEPTH
        return max(1, min(depth, 4))

    def negamax(
        self,
        board: board.Board,
        depth: int,
        alpha: float,
        beta: float,
        time_left: Callable,
    ) -> float:
        if time_left() < 0.05:
            return self.evaluate(board)

        if depth == 0 or board.is_game_over():
            return self.evaluate(board)

        moves = board.get_valid_moves()
        if not moves:
            return -1e6  # no moves = basically lost

        best = -math.inf
        for move in self.order_moves(board, moves):
            if time_left() < 0.02:
                break
            child = self.simulate_move(board, move)
            score = -self.negamax(child, depth - 1, -beta, -alpha, time_left)
            if score > best:
                best = score
            if best > alpha:
                alpha = best
            if alpha >= beta:
                break
        return best

    def simulate_move(self, board: board.Board, move: Tuple[Direction, MoveType]) -> board.Board:
        """
        Use forecast_move to get the next game state.
        We trust the Board to track whose turn it is; negamax flips the sign.
        """
        direction, movetype = move
        new_board: board.Board = board.forecast_move(direction, movetype)
        return new_board

    def order_moves(self, board: board.Board, moves):
        """
        Order moves with a 1-ply quick eval, but strongly prefer moves that lay eggs.
        """
        scored = []
        for mv in moves:
            direction, movetype = mv
            child = self.simulate_move(board, mv)
            h = self.quick_evaluate(child)

            # Primary key: whether this move lays an egg.
            # 1 for egg, 0 otherwise.
            is_egg = 1 if movetype == MoveType.EGG else 0

            # We sort descending on (is_egg, h)
            scored.append(((is_egg, h), mv))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [mv for _, mv in scored]


    def choose_greedy(self, board: board.Board, moves):
        best_mv = None
        best_score = -math.inf
        for mv in moves:
            child = self.simulate_move(board, mv)
            score = self.quick_evaluate(child)
            if score > best_score or best_mv is None:
                best_score = score
                best_mv = mv
        return best_mv if best_mv is not None else moves[0]

    # ================= EVALUATION ================= #

    def evaluate(self, board: board.Board) -> float:
        my_pos = board.chicken_player.get_location()
        opp_pos = board.chicken_enemy.get_location()

        my_egg_score, opp_egg_score = self.current_egg_scores(board)
        my_egg_score = min(my_egg_score, float(self.FIXED_TARGET_EGGS))
        opp_egg_score = min(opp_egg_score, float(self.FIXED_TARGET_EGGS))
        eggs_diff = my_egg_score - opp_egg_score

        my_moves = len(board.get_valid_moves())
        center_term = self.center_weight_grid[my_pos] - self.center_weight_grid[opp_pos]

        pot_my, pot_opp = self.potential_eggs(board, my_pos, opp_pos)
        pot_my = min(pot_my, float(self.FIXED_TARGET_EGGS))
        pot_opp = min(pot_opp, float(self.FIXED_TARGET_EGGS))
        pot_diff = pot_my - pot_opp

        # Trapdoor hard avoidance
        trap_prob_here = self.trap_belief.prob_at(my_pos)
        if trap_prob_here > self.TRAPDOOR_HARD_THRESHOLD:
            return -1e9

        trap_penalty = 0.0
        if trap_prob_here > 0.0:
            trap_penalty = self.TRAPDOOR_WEIGHT * (
                trap_prob_here / (1.0 - trap_prob_here + 1e-6)
            )

        score = 0.0
        score += self.EGGS_WEIGHT * eggs_diff
        score += self.POTENTIAL_WEIGHT * pot_diff
        score += self.MOBILITY_WEIGHT * my_moves
        score += self.CENTER_WEIGHT * center_term
        score -= trap_penalty

        return score



    def quick_evaluate(self, board: board.Board) -> float:
        """
        Cheaper eval used for move ordering and low-time fallbacks.
        Heavily biased towards immediate egg lead and not stepping on trapdoors.
        """
        my_pos = board.chicken_player.get_location()
        my_egg_score, opp_egg_score = self.current_egg_scores(board)
        eggs_diff = my_egg_score - opp_egg_score

        # Small center preference, purely tie-breaky
        center_term = 0.1 * self.center_weight_grid[my_pos]

        trap_prob_here = self.trap_belief.prob_at(my_pos)
        if trap_prob_here > self.TRAPDOOR_HARD_THRESHOLD:
            # Standing on what we think is a trapdoor is just awful.
            return -1e8

        trap_penalty = 0.0
        if trap_prob_here > 0.0:
            trap_penalty = self.TRAPDOOR_WEIGHT * (
                trap_prob_here / (1.0 - trap_prob_here + 1e-6)
            )

        # Eggs drive everything; center is tiny; trap penalty dominates if high.
        return 5.0 * eggs_diff + center_term - trap_penalty



    def current_egg_scores(self, board: board.Board) -> Tuple[float, float]:
        """
        Score current eggs using eggs_player / eggs_enemy sets, plus extra bonus
        for corner eggs (worth 3 total instead of 1).
        """
        eggs_player = getattr(board, "eggs_player", set())
        eggs_enemy = getattr(board, "eggs_enemy", set())

        corners = {
            (0, 0),
            (0, self.map_size - 1),
            (self.map_size - 1, 0),
            (self.map_size - 1, self.map_size - 1),
        }

        def score(eggs_set):
            base = len(eggs_set)
            corner_bonus = sum(1 for e in eggs_set if e in corners) * 2  # +2 on top of base 1
            return float(base + corner_bonus)

        return score(eggs_player), score(eggs_enemy)

    def potential_eggs(
        self,
        board: board.Board,
        my_pos: Tuple[int, int],
        opp_pos: Tuple[int, int],
    ) -> Tuple[float, float]:
        eggs_player = getattr(board, "eggs_player", set())
        eggs_enemy = getattr(board, "eggs_enemy", set())
        turds_player = getattr(board, "turds_player", set())
        turds_enemy = getattr(board, "turds_enemy", set())

        occupied = eggs_player | eggs_enemy | turds_player | turds_enemy

        corners = {
            (0, 0),
            (0, self.map_size - 1),
            (self.map_size - 1, 0),
            (self.map_size - 1, self.map_size - 1),
        }

        pot_my = 0.0
        pot_opp = 0.0

        for x in range(self.map_size):
            for y in range(self.map_size):
                pos = (x, y)
                if pos in occupied:
                    continue

                trap_p = self.trap_belief.prob_at(pos)

                # If this square is very likely a trapdoor, pretend it's worthless.
                if trap_p > self.TRAPDOOR_HARD_THRESHOLD:
                    continue

                # Otherwise discount its value by (1 - k * p), clipped at 0.
                # So even moderate probability makes it less attractive.
                discount = max(0.0, 1.0 - 3.0 * trap_p)

                base_value = 3.0 if pos in corners else 1.0
                value = base_value * discount
                if value <= 0.0:
                    continue

                d_my = self.manhattan_dist(my_pos, pos)
                d_opp = self.manhattan_dist(opp_pos, pos)

                if d_my < d_opp:
                    pot_my += value / (1.0 + d_my)
                elif d_opp < d_my:
                    pot_opp += value / (1.0 + d_opp)
                else:
                    v = value / (1.0 + d_my)
                    pot_my += 0.5 * v
                    pot_opp += 0.5 * v

        return pot_my, pot_opp


    # ================= UTIL ================= #

    @staticmethod
    def manhattan_dist(a, b) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
