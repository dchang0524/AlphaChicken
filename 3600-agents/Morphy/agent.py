from collections.abc import Callable
from typing import List, Tuple
import math

import numpy as np
from game import board  # type: ignore
from game.enums import MoveType, Direction, loc_after_direction  # type: ignore
from .hiddenMarkov import TrapdoorBelief


INF = 10**9


class PlayerAgent:
    def __init__(self, board: board.Board, time_left: Callable):
        self.map_size = board.game_map.MAP_SIZE
        self.trap_belief = TrapdoorBelief(self.map_size)
        self.known_traps: set[Tuple[int, int]] = set()
        #anti repetition
        self.prev_pos: Tuple[int, int] | None = None
        self.prev_opp_eggs: int = 0

        # You can tune these:
        self.max_depth = 9     # typical; drop to 2 if time is low
        self.trap_hard = 0.95   # hard “lava” threshold
        self.trap_weight = 50 # soft risk penalty scale
        self.look_radius = 3 #for heauristic evalutions

    # ------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------
    def play(
        self,
        board: board.Board,
        sensor_data: List[Tuple[bool, bool]],
        time_left: Callable,
    ):
        # 1) Collapse beliefs on any trapdoors the engine has discovered
        # Assuming board.found_trapdoors is a set of (x, y) tuples
        found = getattr(board, "found_trapdoors", set())
        # Just in case it's some other iterable:
        found = set(found)

        new_traps = found - self.known_traps
        for pos in new_traps:
            self.trap_belief.set_trapdoor(pos)
        self.known_traps |= new_traps

        # 2) Normal HMM update with current senses
        my_pos = board.chicken_player.get_location()
        opp_eggs = board.eggs_enemy

        self.trap_belief.update(my_pos, sensor_data)

        depth = self.choose_depth(time_left)

        moves = board.get_valid_moves()
        if not moves:
            return (Direction.STAY, MoveType.MOVE)

        best_move = None
        best_val = -INF

        ordered = self.order_moves(board, moves)

        alpha, beta = -INF, INF
        for mv in ordered:
            child = self.simulate_move(board, mv)
            val = self.alphabeta(
                child, depth - 1, alpha, beta, maximizing=False, time_left=time_left
            )
            if val > best_val:
                best_val = val
                best_move = mv
            alpha = max(alpha, best_val)
            if beta <= alpha:
                break
        self.prev_pos = my_pos
        self.prev_opp_eggs = opp_eggs
        return best_move
    
    def choose_depth(self, time_left):
        """Choose search depth based on remaining time."""
        t = time_left()

        # Emergency mode: barely any time left
        if t < 0.4:
            return 1
        
        # Medium time: slightly reduced search
        if t < 1.5:
            return 2
        
        if t < 30:
            return 4

        if t < 300:
            return 8
        
        # Normal mode (full depth)
        return self.max_depth


    def alphabeta(
        self,
        board: board.Board,
        depth: int,
        alpha: float,
        beta: float,
        maximizing: bool,
        time_left: Callable,
    ) -> float:
        # Fail-safe: if we’re almost out of time, cut search
        if time_left() < 0.05:
            return self.evaluate(board)

        if depth == 0:
            return self.evaluate(board)

        moves = board.get_valid_moves()
        if not moves:
            return self.evaluate(board)

        moves = self.order_moves(board, moves)

        if maximizing:
            value = -INF
            for mv in moves:
                child = self.simulate_move(board, mv)
                value = max(
                    value,
                    self.alphabeta(child, depth - 1, alpha, beta, False, time_left),
                )
                alpha = max(alpha, value)
                if beta <= alpha:
                    break
            return value
        else:
            value = INF
            for mv in moves:
                child = self.simulate_move(board, mv)
                value = min(
                    value,
                    self.alphabeta(child, depth - 1, alpha, beta, True, time_left),
                )
                beta = min(beta, value)
                if beta <= alpha:
                    break
            return value

    def simulate_move(self, b: board.Board, mv) -> board.Board:
        direction, movetype = mv

        # Make a copy of the board, same way the engine does for players
        b2 = b.get_copy(build_history=False, asymmetric=True)

        # Apply the move on the copy.
        # apply_move signature: (dir, move_type, timer=0, check_ok=True)
        ok = b2.apply_move(direction, movetype, timer=0.0, check_ok=True)

        # If something weird happens (shouldn’t, since we use get_valid_moves),
        # just return the copy as-is.
        if not ok:
            return b2

        return b2


    def manhattan(self, a, b) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def evaluate(self, board: board.Board) -> float:
        my_pos = board.chicken_player.get_location()
        opp_pos = board.chicken_enemy.get_location()

        my_eggs = board.eggs_player
        opp_eggs = board.eggs_enemy
        my_turds = board.turds_player
        opp_turds = board.turds_enemy

        base_me = len(my_eggs)
        base_opp = len(opp_eggs)

        moves_left = board.MAX_TURNS - board.turn_count

        # Collect unclaimed squares
        unclaimed = []
        dim = board.game_map.MAP_SIZE
        occupied = my_eggs | opp_eggs | my_turds | opp_turds
        for x in range(dim):
            for y in range(dim):
                if (x, y) not in occupied:
                    unclaimed.append((x, y))

        exp_me = base_me
        exp_opp = base_opp

        for s in unclaimed:
            d_me = self.manhattan(my_pos, s)
            d_opp = self.manhattan(opp_pos, s)

            if d_me > moves_left and d_opp > moves_left:
                continue

            w_me = w_opp = 0.0
            if d_me <= moves_left and d_opp > moves_left:
                w_me = 1.0
            elif d_opp <= moves_left and d_me > moves_left:
                w_opp = 1.0
            else:
                if d_me < d_opp:
                    w_me = 1.0
                elif d_opp < d_me:
                    w_opp = 1.0
                else:
                    w_me = w_opp = 0.5

            # Trap risk discount on this egg
            trap_p = self.trap_belief.prob_at(s)
            safe_factor = 1.0 - trap_p  # brutal: if high trap prob, value ~ 0
            exp_me += w_me * safe_factor
            exp_opp += w_opp  # opponent “doesn’t know”, but fine as heuristic

        # Hard penalty for standing on / moving into likely trap
        trap_here = self.trap_belief.prob_at(my_pos)
        if trap_here > self.trap_hard:
            return -INF / 2

        trap_penalty = self.trap_weight * (trap_here / (1.0 - trap_here + 1e-9))

        return (exp_me - exp_opp) - trap_penalty

    def move_priority(self, mv) -> int:
        direction, movetype = mv
        if movetype == MoveType.EGG:
            return 2
        if movetype == MoveType.TURD:
            return 1
        return 0

    def direction_score(self, cur_board: board.Board, mv) -> float:
        direction, _ = mv
        my_pos = cur_board.chicken_player.get_location()
        opp_pos = cur_board.chicken_enemy.get_location()

        new_pos = loc_after_direction(my_pos, direction)

        eggs_player = cur_board.eggs_player
        eggs_enemy = cur_board.eggs_enemy
        turds_player = cur_board.turds_player
        turds_enemy = cur_board.turds_enemy
        occupied = eggs_player | eggs_enemy | turds_player | turds_enemy

        dim = cur_board.game_map.MAP_SIZE
        score = 0.0

        # Strong penalty for just moving back to where we were
        if self.prev_pos is not None and new_pos == self.prev_pos:
            score -= 5.0  # tune this; big enough to avoid unless truly necessary

        for x in range(dim):
            for y in range(dim):
                pos = (x, y)
                if pos in occupied:
                    continue

                if self.manhattan(new_pos, pos) > self.look_radius:
                    continue

                d_me = self.manhattan(new_pos, pos)
                d_opp = self.manhattan(opp_pos, pos)
                if d_me < d_opp:
                    score += 1.0

        return score



    def order_moves(self, board: board.Board, moves):
        scored = []
        for mv in moves:
            pri = self.move_priority(mv)
            dir_score = self.direction_score(board, mv)
            scored.append(((pri, dir_score), mv))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [mv for _, mv in scored]
