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
        self.trap_hard = 0.5   # hard “lava” threshold
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
            return -INF

        ordered = self.order_moves(board, moves, blocked_dir=None)

        alpha, beta = -INF, INF
        best_val = -INF
        best_move = None


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

        for mv in ordered:
            child = self.simulate_move(board, mv)
            child_blocked = self.opposite(mv[0])

            # Now it's opponent's POV
            child.reverse_perspective()

            # Root negamax call (note the sign flip and window swap)
            val = -self.negamax(child,
                                depth - 1,
                                -beta,
                                -alpha,
                                time_left,
                                blocked_dir=child_blocked)

            if val > best_val:
                best_val = val
                best_move = mv

            if best_val > alpha:
                alpha = best_val
            if alpha >= beta:
                break

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


    def negamax(self,
            board: board.Board,
            depth: int,
            alpha: float,
            beta: float,
            time_left: Callable,
            blocked_dir=None) -> float:
        """
        Negamax on the asymmetric POV board.

        Invariant:
        - At every node, `board` is from the POV of the *current player*.
        - `evaluate(board)` returns score from the POV of the current player.
        - We flip POV with board.reverse_perspective() after every move.
        - We negate the returned value when unwinding recursion.

        value(position for current player) = - value(position for opponent)
        """

        # Hard time cutoff
        if time_left() < 0.05:
            return self.evaluate(board)

        if depth == 0:
            return self.evaluate(board)

        moves = board.get_valid_moves()
        if not moves:
            # No moves for the current POV → they lose
            return -INF

        # You can still use blocked_dir if you want, or ignore it
        moves = self.order_moves(board, moves, blocked_dir)

        best = -INF

        for mv in moves:
            child = self.simulate_move(board, mv)
            child_blocked = self.opposite(mv[0])

            # Now it's opponent's turn → flip POV
            child.reverse_perspective()

            # Negamax step: opponent's best is negative for us
            score = -self.negamax(child,
                                depth - 1,
                                -beta,
                                -alpha,
                                time_left,
                                blocked_dir=child_blocked)

            if score > best:
                best = score
            if best > alpha:
                alpha = best
            if alpha >= beta:
                break

        return best


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
        my_pos = cur_board.chicken_player.get_location()
        opp_pos = cur_board.chicken_enemy.get_location()  # currently unused, but keep if you want later

        my_eggs = cur_board.eggs_player
        opp_eggs = cur_board.eggs_enemy
        my_turds = cur_board.turds_player
        opp_turds = cur_board.turds_enemy

        base_me = len(my_eggs)
        base_opp = len(opp_eggs)

        moves_left = cur_board.MAX_TURNS - cur_board.turn_count
        dim = cur_board.game_map.MAP_SIZE

        # If no moves left, just return material + trap penalty at current location.
        trap_here = self.trap_belief.prob_at(my_pos)
        trap_penalty_here = 10.0 * trap_here
        if moves_left <= 0:
            return (base_me - base_opp) - trap_penalty_here

        # --- d = sqrt(moves_left), clamped to [1, dim] ---
        d = int(moves_left ** 0.5)
        if d < 1:
            d = 1
        if d > dim:
            d = dim

        occupied = my_eggs | opp_eggs | my_turds | opp_turds

        mx, my = my_pos

        # Candidate windows: treat our position as each corner of the dxd square,
        # then clamp to board. Deduplicate resulting (x0, y0).
        candidate_windows = set()
        corner_offsets = [
            (0, 0),          # our pos at top-left of window
            (d - 1, 0),      # our pos at top-right
            (0, d - 1),      # bottom-left
            (d - 1, d - 1),  # bottom-right
        ]

        for offx, offy in corner_offsets:
            x0 = mx - offx
            y0 = my - offy
            # clamp so window stays on board
            if x0 < 0:
                x0 = 0
            if y0 < 0:
                y0 = 0
            if x0 > dim - d:
                x0 = dim - d
            if y0 > dim - d:
                y0 = dim - d
            candidate_windows.add((x0, y0))

        # Opponent future placement penalty:
        region_area = d * d
        opp_future_factor = (moves_left / 4.0) * (region_area / float(dim * dim))

        best_region_score = float("-inf")

        for x0, y0 in candidate_windows:
            unclaimed = 0
            opp_eggs_count = 0
            opp_turds_count = 0
            trap_prob_sum = 0.0
            newly_visited = 0
            for x in range(x0, x0 + d):
                for y in range(y0, y0 + d):
                    s = (x, y)

                    # "Unclaimed eggs": squares not occupied by any egg or turd
                    if s not in occupied:
                        unclaimed += 1

                    if s in opp_eggs:
                        opp_eggs_count += 1
                    if s in opp_turds:
                        opp_turds_count += 1
                    if not self.visited[x][y]:
                        newly_visited += 1

                    

                    trap_prob_sum += self.trap_belief.prob_at(s)

            # Your heuristic:
            # score_region = (#unclaimed)
            #                - 0.5 * (#opp eggs)
            #                - 1.0 * (#opp turds)
            #                - 5.0 * (sum trap probabilities in region)
            #                - opp_future_factor (optional)
            score_region = (
                float(unclaimed)
                + 0.5 * newly_visited
                - 0.5 * opp_eggs_count
                - 1.0 * opp_turds_count
                - 5.0 * trap_prob_sum
                - opp_future_factor
            )

            if score_region > best_region_score:
                best_region_score = score_region

        # If something went very wrong and we never set best_region_score, fall back.
        if best_region_score == float("-inf"):
            future_gain = 0.0
        else:
            future_gain = best_region_score
        center_bonus = self.space_advantage(my_eggs, my_turds, cur_board.turn_count)
        # Combine: real eggs + discounted expected future eggs - trap risk where we stand.
        base_diff = base_me - base_opp
        score = base_diff + 0.5 * (future_gain - trap_penalty_here - opp_future_factor) - 1.3*trap_penalty_here + center_bonus

        return score
    
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

        # 2. Remove moves stepping on known / high-prob trapdoors
        filtered: list[tuple[Direction, MoveType]] = []
        for direction, movetype in moves:
            next_loc = loc_after_direction(cur_loc, direction)
            # Never step on discovered trapdoors
            if next_loc in self.known_traps:
                continue
            # Block squares whose trapdoor probability is too high
            trap_p = self.trap_belief.prob_at(next_loc)
            if trap_p > self.trap_hard:
                continue
            filtered.append((direction, movetype))
        if filtered:
            moves = filtered

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
