from collections.abc import Callable
from collections import deque
from typing import List, Tuple
import math

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
        
        #anti repetition
        self.prev_pos: Tuple[int, int] | None = None
        self.visited = [[False for _ in range(8)] for _ in range(8)]

        # Hyperparameters:
        self.max_depth = 10     # typical; drop to 2 if time is low
        self.trap_hard = 0.95   # hard “lava” threshold
        self.trap_weight = 50 # soft risk penalty scale
        self.look_radius = 3 #for heauristic evalutions

    def play(
        self,
        board: board.Board,
        sensor_data: List[Tuple[bool, bool]],
        time_left: Callable,
    ):
        x, y = board.chicken_player.get_location()
        if 0 <= x < len(self.visited) and 0 <= y < len(self.visited[0]):
            self.visited[x][y] = True
        # 1) Collapse beliefs on any trapdoors the engine has discovered
        found = board.found_trapdoors 
        new_traps = found - self.known_traps
        for pos in new_traps:
            self.trap_belief.set_trapdoor(pos)
        self.known_traps |= new_traps

        # 2) Normal HMM update with current senses
        my_pos = board.chicken_player.get_location()

        self.trap_belief.update(my_pos, sensor_data)

        #Alpha-Beta Search for best move
        #TODO: Expected Value- add in a hard prediction mechanism?

        depth = self.choose_depth(board.player_time, board.turn_count)
        moves = board.get_valid_moves()
        if not moves:
            return (Direction.STAY, MoveType.MOVE)

        best_move = None
        best_val = -INF

        ordered = self.order_moves(board, moves, blocked_dir=None)
        # --- Root-level backtracking prevention ---
        if self.prev_pos is not None:
            cur_loc = board.chicken_player.get_location()
            filtered = []
            for direction, movetype in ordered:
                next_loc = loc_after_direction(cur_loc, direction)
                if next_loc != self.prev_pos:
                    filtered.append((direction, movetype))
            # Only override if we didn't kill everything
            if filtered:
                ordered = filtered

        alpha, beta = -INF, INF
        for mv in ordered:
            child = self.simulate_move(board, mv)
            child_blocked = self.opposite(mv[0])
            val = self.alphabeta(
                child, depth - 1, alpha, beta, maximizing=False, time_left=time_left, blocked_dir=child_blocked
            )
            if val > best_val:
                best_val = val
                best_move = mv
            alpha = max(alpha, best_val)
            if beta <= alpha:
                break
        self.prev_pos = my_pos
        return best_move
    
    def choose_depth(self, time_left, turn):
        """Choose search depth based on remaining time."""
        t = time_left
        #TODO: In the opening stage, don't waste too much time
        #TODO: Variable Depth, depending on how complex the position is

        #if turn < 4:
            #return 5

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
        if t < 180:
            return self.max_depth - 1
        
        # Normal mode (full depth)
        return self.max_depth


    def alphabeta(self, board, depth, alpha, beta, maximizing, time_left, blocked_dir):
        # Fail-safe: if we’re almost out of time, cut search
        if time_left() < 0.05:
            return self.evaluate(board)

        if depth == 0:
            return self.evaluate(board)

        moves = board.get_valid_moves()
        if not moves: #no moves left
            # 'maximizing' == True  → it's our move → we lose
            # 'maximizing' == False → it's opponent's move → they lose → we win
            return -INF if maximizing else INF

        moves = self.order_moves(board, moves, blocked_dir)

        if maximizing:
            value = -INF
            for mv in moves:
                child = self.simulate_move(board, mv)
                child_blocked = self.opposite(mv[0])

                value = max(
                    value,
                    self.alphabeta(child, depth - 1, alpha, beta, False, time_left, blocked_dir=child_blocked),
                )
                alpha = max(alpha, value)
                if beta <= alpha:
                    break
            return value
        else:
            value = INF
            for mv in moves:
                child = self.simulate_move(board, mv)
                child_blocked = self.opposite(mv[0])
                value = min(
                    value,
                    self.alphabeta(child, depth - 1, alpha, beta, True, time_left, blocked_dir=child_blocked),
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
    
    def opposite(self, d: Direction) -> Direction:
        match d:
            case Direction.UP: return Direction.DOWN
            case Direction.DOWN: return Direction.UP
            case Direction.LEFT: return Direction.RIGHT
            case Direction.RIGHT: return Direction.LEFT
            case _: return d


    #TODO: Consider adding "momentum" metric, where if you pick a direction you keep going at it
    def evaluate(self, cur_board: board.Board) -> float:
        """
        Evaluation using:
        - Egg difference (current eggs on board)
        - Voronoi-style egg space control (space_control)
        - Local trapdoor risk
        - Center-breakthrough bonus with time decay
        """

        my_eggs  = cur_board.eggs_player
        opp_eggs = cur_board.eggs_enemy

        base_me   = len(my_eggs)
        base_opp  = len(opp_eggs)
        base_diff = base_me - base_opp

        # Voronoi egg-space control
        my_space, opp_space = self.space_control(cur_board)
        space_gap = my_space - opp_space

        # Local trap risk
        my_pos = cur_board.chicken_player.get_location()
        trap_here = self.trap_belief.prob_at(my_pos)
        TRAP_LOCAL_COST = 10.0
        trap_penalty = TRAP_LOCAL_COST * trap_here

        # Time-decay factor: early turns get ~1, late turns ~0
        max_turns = cur_board.MAX_TURNS
        t = cur_board.turn_count
        if max_turns > 0:
            decay = max(0.0, (max_turns - t) / max_turns)
        else:
            decay = 0.0

        # Center-breakthrough bonus:
        # count our eggs/turds that lie past the frontier
        center_adv_count = 0
        for pos in my_eggs | cur_board.turds_player:
            if self.is_forward_square(pos):
                center_adv_count += 1

        CENTER_WEIGHT = 3.0  # tuneable
        center_bonus = CENTER_WEIGHT * center_adv_count * decay

        # Phase logic driven purely by opponent egg-space
        FINISH_THRESHOLD = 10   # they control very few egg-eligible squares
        CRUSH_THRESHOLD  = 26   # clearly cramped but not dead

        # a) Finish mode: low opp space -> convert
        if opp_space <= FINISH_THRESHOLD:
            return (
                5.0 * base_diff
                + 4.0 * space_gap
                + 2.0 * center_bonus
                - 1.0 * trap_penalty
            )

        # b) Crush mode: medium opp space
        if opp_space <= CRUSH_THRESHOLD:
            return (
                4.0 * space_gap
                + 3.0 * base_diff
                + 1.5 * center_bonus
                - 1.0 * trap_penalty
            )

        # c) Full Tal mode: they still have a lot of egg-space
        return (
            6.0 * space_gap        # dominate future egg territory
            + 1.0 * base_diff      # eggs secondary
            + 2.0 * center_bonus   # strong incentive to break through early
            - 1.2 * trap_penalty
        )

    
    def space_advantage(self, my_eggs, my_turds, turn: int) -> float:
        """
        Rewards for board control:
        - Real center 2×2: eggs +1.0, turds +0.8
        - Center 4×4: eggs +0.5, turds +0.5
        - Corner squares 1×1: eggs +2.0

        In early-mid game (turn <= 15), center control
        (2×2 + 4×4) gets an extra multiplier.
        """

        # Real center (2×2)
        real_center = {(3, 3), (3, 4), (4, 3), (4, 4)}

        # Center (4×4)
        center_rows = {2, 3, 4, 5}
        center_cols = {2, 3, 4, 5}

        # Corners (8×8 board)
        corners = {(0, 0), (0, 7), (7, 0), (7, 7)}

        bonus = 0.0

        # Early-mid game: boost center control
        center_mult = 2 if turn <= 15 else 1.0

        # Eggs
        for (x, y) in my_eggs:
            if (x, y) in corners:
                # Corner bonus is from game rules, don't scale with time
                bonus += 2.0
            elif (x, y) in real_center:
                bonus += center_mult * 2.0
            elif x in center_rows and y in center_cols:
                bonus += center_mult * 1

        # Turds
        for (x, y) in my_turds:
            if (x, y) in real_center:
                bonus += center_mult * 1.5
            elif x in center_rows and y in center_cols:
                bonus += center_mult * 1

        return bonus
    
    def is_forward_square(self, pos: tuple[int, int]) -> bool:
        """
        Returns True if `pos` is on the 'far side' of the frontier
        relative to our spawn position.

        Frontier is defined by x=3/4 and y=3/4 splits on an 8×8 board.
        """
        sx, sy = self.spawn_pos
        x, y = pos

        # top-left spawn: [0..3]x[0..3]
        if sx <= 3 and sy <= 3:
            return (x >= 4) or (y >= 4)

        # top-right spawn: [0..3]x[4..7]
        if sx <= 3 and sy >= 4:
            return (x >= 4) or (y <= 3)

        # bottom-left spawn: [4..7]x[0..3]
        if sx >= 4 and sy <= 3:
            return (x <= 3) or (y >= 4)

        # bottom-right spawn: [4..7]x[4..7]
        return (x <= 3) or (y <= 3)




    def order_moves(self, board: board.Board, moves, blocked_dir=None):
        cur_loc = board.chicken_player.get_location()
        dim = board.game_map.MAP_SIZE
        x, y = cur_loc

        # --- geometric classification -----------------------------------------
        # Corner region: outer 2 rows or outer 2 columns
        is_corner = (x < 2 or x >= dim - 2) or (y < 2 or y >= dim - 2)

        # Center region: inner 4x4 (rows 2..5, cols 2..5)
        is_center = (2 <= x <= 5) and (2 <= y <= 5)

        # ---------------------------------------------------------------------

        # 1. Egg filtering
        egg_moves = [mv for mv in moves if mv[1] == MoveType.EGG]

        if egg_moves and not is_center:
            # In non-center regions: only egg moves are kept
            moves = egg_moves
        # In center: keep all moves; eggs still sort first later

        # 1.5 Prevent immediate backtracking
        if blocked_dir is not None:
            non_backtracking = []
            for direction, movetype in moves:
                if direction != blocked_dir:
                    non_backtracking.append((direction, movetype))
            if non_backtracking:
                moves = non_backtracking

        # 2. Remove moves stepping on known trapdoors
        safe_moves = []
        for direction, movetype in moves:
            next_loc = loc_after_direction(cur_loc, direction)
            if next_loc not in self.known_traps:
                safe_moves.append((direction, movetype))
        if safe_moves:
            moves = safe_moves

        # 3. Score moves cheaply
        scored = []
        for mv in moves:
            direction, movetype = mv

            # priority: EGG > TURD > MOVE (default)
            pri = self.move_priority(mv)

            # Corner rule: penalize turds relative to MOVE
            if is_corner and movetype == MoveType.TURD:
                pri = -1     # lower than MOVE = 0

            dir_score = self.direction_score(board, mv)

            scored.append((pri, dir_score, mv))

        # 4. Sort by (priority, direction_score)
        scored.sort(key=lambda t: (t[0], t[1]), reverse=True)

        return [t[2] for t in scored]



    def move_priority(self, mv) -> int:
        direction, movetype = mv
        if movetype == MoveType.EGG:
            return 2
        if movetype == MoveType.TURD:
            return 1
        return 0

    def direction_score(self, cur_board: board.Board, mv) -> float:
        """
        Directional heuristic: for the chosen direction, look at the
        half-board region in that direction from our current position.

        Score(region) =
            (# of unlaid squares)
        - 0.5 * (# of opponent eggs)
        - 1.0 * (# of opponent turds)
        - 10.0 * (sum of trapdoor probabilities in region)
        """
        direction, _ = mv
        dim = cur_board.game_map.MAP_SIZE

        my_pos = cur_board.chicken_player.get_location()
        mx, my = my_pos

        eggs_player = cur_board.eggs_player
        eggs_enemy = cur_board.eggs_enemy
        turds_player = cur_board.turds_player
        turds_enemy = cur_board.turds_enemy
        occupied = eggs_player | eggs_enemy | turds_player | turds_enemy

        # Define the region for this direction
        def in_region(x: int, y: int) -> bool:
            if direction == Direction.UP:
                # your example: at (3,4), UP → y > 4
                return y > my
            elif direction == Direction.DOWN:
                return y < my
            elif direction == Direction.RIGHT:
                return x > mx
            elif direction == Direction.LEFT:
                return x < mx
            else:
                # STAY or any other direction: no clear "front" region,
                # just don't bias it (or treat the whole board as region).
                return True

        unlaid = 0
        opp_eggs_count = 0
        opp_turds_count = 0
        trap_prob_sum = 0.0

        for x in range(dim):
            for y in range(dim):
                if not in_region(x, y):
                    continue

                pos = (x, y)

                # unlaid = not occupied by any egg or turd
                if pos not in occupied:
                    unlaid += 1

                if pos in eggs_enemy:
                    opp_eggs_count += 1
                if pos in turds_enemy:
                    opp_turds_count += 1

                trap_prob_sum += self.trap_belief.prob_at(pos)

        score = (
            float(unlaid)
            - 0.5 * opp_eggs_count
            - 1.0 * opp_turds_count
            - 10.0 * trap_prob_sum
        )

        return score



    #too slow to actually use for evalution
    def bfs_distances(self, cur_board: board.Board, start: tuple[int, int], for_me: bool) -> dict[tuple[int, int], int]:
        """
        BFS over the board respecting movement constraints.

        for_me = True  -> distances for our chicken
        for_me = False -> distances for opponent chicken
        """
        dim = cur_board.game_map.MAP_SIZE

        eggs_p = cur_board.eggs_player
        eggs_e = cur_board.eggs_enemy
        turds_p = cur_board.turds_player
        turds_e = cur_board.turds_enemy

        # Squares you simply cannot step on (anyone)
        blocked = eggs_p | eggs_e | turds_p | turds_e

        # Squares forbidden because they share an edge with *opponent* turds
        # For us -> forbidden around opponent's turds
        # For opponent -> forbidden around our turds
        if for_me:
            opp_turds = turds_e
        else:
            opp_turds = turds_p

        forbidden_adjacent: set[tuple[int, int]] = set()
        for (tx, ty) in opp_turds:
            for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
                nx, ny = tx + dx, ty + dy
                if 0 <= nx < dim and 0 <= ny < dim:
                    forbidden_adjacent.add((nx, ny))

        # We also never step on the trapdoor squares themselves once known,
        # but that's already encoded via board.turds if they place turds there.
        # If you want, you can additionally block squares in board.found_trapdoors.

        dist: dict[tuple[int, int], int] = {}
        q = deque()

        if start in blocked or start in forbidden_adjacent:
            return dist  # we are in a horrible position; everything is unreachable

        dist[start] = 0
        q.append(start)

        while q:
            x, y = q.popleft()
            d = dist[(x, y)]

            for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
                nx, ny = x + dx, y + dy
                if not (0 <= nx < dim and 0 <= ny < dim):
                    continue
                pos = (nx, ny)
                if pos in dist:
                    continue
                if pos in blocked:
                    continue
                if pos in forbidden_adjacent:
                    continue

                dist[pos] = d + 1
                q.append(pos)

        return dist
