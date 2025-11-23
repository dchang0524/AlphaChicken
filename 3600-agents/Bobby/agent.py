from collections.abc import Callable
from collections import deque
from typing import List, Tuple
import math
import os

import numpy as np
from game import board  # type: ignore
from game.enums import MoveType, Direction, loc_after_direction  # type: ignore
from .hiddenMarkov import TrapdoorBelief


INF = 10**9


class PlayerAgent:
    def __init__(self, board: board.Board, time_left: Callable):
        #board & trapdoor details
        self.map_size = board.game_map.MAP_SIZE
        self.trap_belief = TrapdoorBelief(self.map_size)
        self.known_traps: set[Tuple[int, int]] = set()
        self.spawn_pos: Tuple[int, int] = board.chicken_player.get_location()
        self.isEvenChicken = board.chicken_player.even_chicken

        #anti repetition
        self.prev_pos: Tuple[int, int] | None = None
        self.visited = [[False for _ in range(8)] for _ in range(8)]

        # Hyperparameters:
        self.max_depth = 7     # typical; drop to 2 if time is low
        self.trap_hard = 0.4   # hard “lava” threshold


    def play(
        self,
        board: board.Board,
        sensor_data: List[Tuple[bool, bool]],
        time_left: Callable,
    ):
        # 1) Collapse beliefs on any trapdoors the engine has discovered
        found = board.found_trapdoors 
        new_traps = found - self.known_traps
        for pos in new_traps:
            self.trap_belief.set_trapdoor(pos)
        self.known_traps |= new_traps

        # 2) Normal HMM update with current senses
        my_pos = board.chicken_player.get_location()
        self.trap_belief.update(my_pos, sensor_data)
        self.trap_belief.mark_safe(my_pos)
        self.trap_belief.mark_safe(board.chicken_enemy.get_location())

        # 3) Record Current State
        moves = board.get_valid_moves()
        self.state = board
        self.dist_player = self.bfs_distances(board, board.chicken_player.get_location(), for_me=True)
        self.dist_enemy = self.bfs_distances(board, board.chicken_enemy.get_location(), for_me=False)

        #Alpha-Beta Search for best move
        depth = self.choose_depth(board.player_time, board.turn_count)
        if not moves:
            return -INF

        ordered = self.order_moves(board, moves, blocked_dir=None)
        alpha, beta = -INF, INF
        best_move = None
        best_val = -INF

        for mv in ordered:
            child = self.simulate_move(board, mv)
            child_blocked = self.opposite(mv[0])  # if you want to use it later

            # Now it's opponent's turn → flip POV
            child.reverse_perspective()

            # Negamax call: note the sign flip and (−beta, −alpha)
            val = -self.negamax(child, depth - 1,
                                -beta, -alpha,
                                time_left,
                                blocked_dir=child_blocked)

            if val > best_val:
                best_val = val
                best_move = mv

            alpha = max(alpha, best_val)
            if alpha >= beta:
                break

        self.prev_pos = my_pos
        return best_move
    
    def choose_depth(self, time_left, turn):
        """Choose search depth based on remaining time."""
        t = time_left
        #TODO: In the opening stage, don't waste too much time
        #TODO: Variable Depth, depending on how complex the position is

        # Emergency mode: barely any time left
        if t < 0.4:
            return 1
        
        # Medium time: slightly reduced search
        if t < 1.5:
            return 2
        if t < 10:
            return 4
        if t < 30:
            return self.max_depth - 3
        if t < 90:
            return self.max_depth - 2
        if t < 150:
            return self.max_depth - 1
        
        # Normal mode (full depth)
        return self.max_depth


    def negamax(self, board: board.Board, depth: int,
            alpha: float, beta: float,
            time_left: Callable,
            blocked_dir=None) -> float:
        """
        Negamax on the asymmetric POV board.

        Invariant:
        - At every node, `board` is from the POV of the *current* player.
        - `evaluate(board)` always returns a score from the POV of the current player.
        - We flip POV with board.reverse_perspective() after every move.
        - We flip the sign of the returned value when we go back up the tree.

        So:
        value(position for current player) = - value(position for opponent)
        """
        #shared evaluation metrics
        # --- Distances for both players (movement constraints respected) ---
        my_pos  = board.chicken_player.get_location()
        opp_pos = board.chicken_enemy.get_location()
        my_dist  = self.bfs_distances(board, my_pos,  for_me=True)
        opp_dist = self.bfs_distances(board, opp_pos, for_me=False)

        # Hard time cutoff
        if time_left() < 0.05 or depth == 0:            
            return self.evaluate(board, my_dist, opp_dist)

        moves = board.get_valid_moves()
        if not moves:
            # Current POV has no moves → they lose badly.
            return -INF

        # You *can* use blocked_dir if you want; for now we ignore it inside tree.
        moves = self.order_moves(board, moves, blocked_dir=None)

        best = -INF

        for mv in moves:
            child = self.simulate_move(board, mv)

            # Now it's the opponent's turn → flip POV
            child.reverse_perspective()

            # Negamax: opponent's best is our worst
            score = -self.negamax(child, depth - 1,
                                -beta, -alpha,
                                time_left,)

            if score > best:
                best = score
            if best > alpha:
                alpha = best
            if alpha >= beta:
                break

        return best
    