from collections.abc import Callable
from time import sleep
from typing import List, Set, Tuple

import numpy as np
from game import *


"""
John's agent to try to understand how the game works.
"""

INF = 10**9
class PlayerAgent:
    """
    /you may add functions, however, __init__ and play are the entry points for
    your program and should not be changed.
    """

    def __init__(self, board: board.Board, time_left: Callable):
        pass

    def play(
        self,
        board: board.Board,
        sensor_data: List[Tuple[bool, bool]],
        time_left: Callable,
    ):  
        unordered_moves = board.get_valid_moves()
        for move in unordered_moves:
            move_type = move[1]
            if move_type == enums.MoveType.EGG:
                return move


        moves = self.order_moves(board, unordered_moves)
        alpha = -INF
        beta = INF
        best_score = -INF
        best_move = moves[0]
        for move in moves:
            dir, move_type = move
            child = board.forecast_move(dir, move_type)
            child.reverse_perspective()
            cur_score = -self.search(child, 6, -beta, -alpha)
            if cur_score >= best_score:
                best_score = cur_score
                best_move = move
            if best_score >= alpha:
                alpha = best_score
            # don't need to check beta becuase first move so opponent hasn't gone

        return best_move

    

    # board is the node/state
    def search(self, board: board.Board, depth: int, alpha: float, beta: float) -> float:
        # minimax (negamax implementation)
        # alpha is the floor (the best eval we've found so far and we want to raise it)
        # beta is the ceiling (the best eval the opponent will allow us to have)
        if depth == 0:
            return self.evaluate(board)
        
        moves = board.get_valid_moves()
        moves = self.order_moves(board, moves)
        if not moves:
            return -INF
        
        best_score = -INF
        for move in moves:
            dir, move_type = move
            child = board.forecast_move(dir, move_type)
            child.reverse_perspective()
            # zero-sum game so take negative of opponents evalution
            # reverse alpha and beta because its now opponents pov. So the negation of our floor will be their ceiling. 
            cur_score = -self.search(child, depth - 1, -beta, -alpha)

            if cur_score >= best_score:
                best_score = cur_score
            
            # this move is too good and opponent has already found a move that is worse for us so they won't let us pick this
            if best_score >= beta:
                return beta
            # update our floor
            if best_score >= alpha:
                alpha = best_score         
        return best_score 


    def evaluate(self, board: board.Board) -> float:
        import random
        
        # 1. INSTANT LOSS CHECK
        if board.chicken_player.get_turds_placed() >= 1:
            return -INF

        score = 0.0

        # 2. THE "DO IT" INCENTIVE (Material)
        # This number is MASSIVE. 
        # It forces the bot to lay the egg even if it means stepping into fire.
        real_egg_count = board.chicken_player.get_eggs_laid()
        score += 10000.0 * real_egg_count 

        # 3. THE "NAVIGATION" INCENTIVE (Parity)
        r, c = board.chicken_player.get_location()
        
        # Is this an EVEN square? (The only place we can act)
        if (r + c) % 2 == 0:
            score += 100.0  # Great, we are on fertile ground.
            
            # Now, are we blocking it?
            # If we are standing on an egg we already laid, we need to leave.
            try:
                # Check if current location is in our egg set
                if hasattr(board, 'player_eggs') and (r, c) in board.player_eggs:
                    score -= 50.0 # Get off the egg! Go find a new +100 square.
            except:
                pass
        else:
            # We are on an ODD square. This is useless dirt.
            score -= 10.0 

        # 4. THE "DONT GET STUCK" INCENTIVE (Mobility)
        # Counts valid moves so we don't trap ourselves
        score += len(board.get_valid_moves()) * 1.0

        # 5. THE "STOP DANCING" INCENTIVE (Noise)
        score += random.random() * 0.5

        return score

    def order_moves(self, board, moves):
        egg_moves = []
        regular_moves = []
        for move in moves:
            move_type = move[1]
            if move_type == enums.MoveType.EGG:
                egg_moves.append(move)
            elif move_type == enums.MoveType.TURD:
                continue
            else:
                regular_moves.append(move)
        return egg_moves + regular_moves



    def manhattan_distance(self, p1: Tuple[int, int], p2: Tuple[int, int]) -> int:
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
