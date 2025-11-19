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
        if t < 150:
            return self.max_depth - 1
        
        # Normal mode (full depth)
        return self.max_depth


    def alphabeta(self, board, depth, alpha, beta, maximizing, time_left, blocked_dir):
        # Fail-safe: if we’re almost out of time, cut search
        if time_left() < 0.05:
            return self.evaluate(board)

        moves = board.get_valid_moves()
        if not moves: #no moves left
            # 'maximizing' == True  → it's our move → we lose
            # 'maximizing' == False → it's opponent's move → they lose → we win
            return -INF if maximizing else INF

        if depth == 0:
            return self.evaluate(board)
        
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


    def evaluate(self, cur_board: board.Board) -> float:
        """
        Tal-style evaluation:

        - PRIMARY: minimize opponent's future egg-path mobility.
        - SECONDARY: center/space control.
        - TERTIARY: raw egg difference.
        - Trap risk is penalized.

        Uses three regimes based on opponent mobility:
        - High mobility     -> pure strangulation focus.
        - Medium mobility   -> strangulation + start cashing egg lead.
        - Very low mobility -> switch to greedy, convert position to egg lead.
        """

        my_pos  = cur_board.chicken_player.get_location()
        my_eggs = cur_board.eggs_player
        opp_eggs = cur_board.eggs_enemy
        my_turds = cur_board.turds_player

        base_me  = len(my_eggs)
        base_opp = len(opp_eggs)
        base_diff = base_me - base_opp

        moves_left = cur_board.MAX_TURNS - cur_board.turn_count

        # Trap penalty at our current square
        trap_here = self.trap_belief.prob_at(my_pos)
        trap_penalty_here = 7.0 * trap_here

        # If no moves left, just return material minus trap risk.
        if moves_left <= 0:
            return base_diff - trap_penalty_here

        # Spatial center control you already defined
        center_bonus = self.space_advantage(my_eggs, my_turds, cur_board.turn_count)
        #center_bonus = 0

        # Opponent egg-path mobility (our primary "Tal pressure" metric)
        opp_mob = self.opp_egg_path_mobility(cur_board)

        # Regime thresholds; tuneable but these are reasonable starting points
        FINISH_THRESHOLD = 8.0   # they basically can't realize more eggs
        CRUSH_THRESHOLD  = 20.0  # they're cramped but not fully dead

        # 1) Endgame / conversion mode:
        # Opponent has almost no egg-paths left. Stop playing Tal, just cash in.
        if opp_mob <= FINISH_THRESHOLD:
            return (
                100.0 * base_diff      # heavily convert egg lead
                + 3.0 * center_bonus
                - 100 * trap_penalty_here
            )

        # 2) Crush mode:
        # They are significantly constrained, but still have some play.
        if opp_mob <= CRUSH_THRESHOLD:
            return (
                -2.5 * opp_mob       # keep strangling their remaining paths
                + 3.0 * base_diff    # but egg lead starts to matter more
                + 3.0 * center_bonus
                - 7 * trap_penalty_here
            )

        # 3) Full Tal mode:
        # Opponent still has healthy mobility. Your job is to murder their graph.
        score = (
            -4.0 * opp_mob          # hard focus: kill their egg mobility
            + 3.0 * center_bonus
            + 0.5 * base_diff       # eggs matter a bit, but not main thing yet
            - 7 * trap_penalty_here
        )
        return score


    def opp_egg_path_mobility(self, cur_board: board.Board) -> float:
        """
        Heuristic 'mobility' for the opponent:
        Sum over all potential opponent egg squares of the (capped) number of
        shortest paths from the opponent's current position.

        - Uses one BFS from opponent's location.
        - Respects movement constraints (eggs, turds, adjacency to *our* turds, chicken).
        - Treats found trapdoors as blocked (stepping on them is effectively losing).
        - For each reachable, empty, parity-correct square, adds min(#paths, PATH_CAP).
        """

        dim = cur_board.game_map.MAP_SIZE

        eggs_p = cur_board.eggs_player
        eggs_e = cur_board.eggs_enemy
        turds_p = cur_board.turds_player
        turds_e = cur_board.turds_enemy

        my_chicken_pos  = cur_board.chicken_player.get_location()
        opp_chicken_pos = cur_board.chicken_enemy.get_location()

        # Squares no one can step on: any egg or turd, plus our chicken.
        blocked: set[tuple[int, int]] = set()
        blocked |= eggs_p
        blocked |= eggs_e
        blocked |= turds_p
        blocked |= turds_e
        blocked.add(my_chicken_pos)

        # Also treat discovered trapdoors as blocked: path through them is suicidal.
        found_traps = getattr(cur_board, "found_trapdoors", set())
        blocked |= set(found_traps)

        # Squares forbidden for the opponent because they share an edge
        # with *our* turds.
        forbidden_adjacent: set[tuple[int, int]] = set()
        for (tx, ty) in turds_p:
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nx, ny = tx + dx, ty + dy
                if 0 <= nx < dim and 0 <= ny < dim:
                    forbidden_adjacent.add((nx, ny))

        start = opp_chicken_pos

        # If the opponent is already in a terrible illegal/blocked spot, treat
        # their future egg mobility as zero.
        if start in blocked or start in forbidden_adjacent:
            return 0.0

        # Parity: (0,0) is even. Opponent parity inferred via (0,1).
        # If we can lay at (0,0), we are even; then opponent is odd -> can_lay at (0,1).
        # If we are odd, then we can't lay at (0,0), but we can at (0,1), so that still
        # gives the opponent's parity correctly.
        opp_even = cur_board.can_lay_egg_at_loc((0, 1))

        from collections import deque

        dist: dict[tuple[int, int], int] = {}
        paths: dict[tuple[int, int], int] = {}

        q = deque()
        dist[start] = 0
        paths[start] = 1
        q.append(start)

        PATH_CAP = 3  # we don't need exact counts; just 0/1/2/3+ notion

        while q:
            x, y = q.popleft()
            d = dist[(x, y)]

            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nx, ny = x + dx, y + dy
                if not (0 <= nx < dim and 0 <= ny < dim):
                    continue
                pos = (nx, ny)

                if pos in blocked or pos in forbidden_adjacent:
                    continue

                if pos not in dist:
                    dist[pos] = d + 1
                    # First time discovered: inherit path count
                    paths[pos] = min(paths[(x, y)], PATH_CAP)
                    q.append(pos)
                elif dist[pos] == d + 1:
                    # Another shortest path found
                    paths[pos] = min(paths[pos] + paths[(x, y)], PATH_CAP)

        # Now sum over all reachable potential egg squares for the opponent.
        total_mobility = 0.0

        for pos, path_count in paths.items():
            # Can't lay eggs on occupied or forbidden squares anyway.
            if pos in blocked:
                continue

            x, y = pos
            even = ((x + y) % 2 == 0)

            # Only count squares of correct parity for opponent egg placement.
            if even != opp_even:
                continue

            # This is a potential egg square: add capped path-count.
            total_mobility += min(path_count, PATH_CAP)

        return total_mobility

    
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

        opp_pos = board.chicken_enemy.get_location()
        dist_to_opp = self.manhattan(cur_loc, opp_pos)

        # --- geometric classification -----------------------------------------
        is_corner = (x < 2 or x >= dim - 2) or (y < 2 or y >= dim - 2)
        is_center = (2 <= x <= 5) and (2 <= y <= 5)

        # "Hot zone" where offensive turds make sense:
        turd_hot_zone = is_center or (dist_to_opp <= 3)

        # Remaining turds (max 5)
        my_turds = board.turds_player
        remaining_turds = max(0, 5 - len(my_turds))

        # Opponent egg-path mobility: how free they still are to grow eggs.
        opp_mob = self.opp_egg_path_mobility(board)
        FINISH_THRESHOLD = 8.0    # basically egg-dead
        CRUSH_THRESHOLD  = 20.0   # significantly cramped

        # ---------------------------------------------------------------------
        # 1. Egg filtering (keep your original behavior)
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

        # 2. Remove moves stepping on known trapdoors (absolute)
        safe_moves = []
        for direction, movetype in moves:
            next_loc = loc_after_direction(cur_loc, direction)
            if next_loc not in self.known_traps:
                safe_moves.append((direction, movetype))
        if safe_moves:
            moves = safe_moves

        # 3. Score moves (Tal-style phases)
        scored: list[tuple[float, float, tuple[Direction, MoveType]]] = []

        # Weight of directional score by phase
        if opp_mob > CRUSH_THRESHOLD:
            dir_w = 1.0   # full Tal: directional space really matters
        elif opp_mob > FINISH_THRESHOLD:
            dir_w = 0.7   # crush mode
        else:
            dir_w = 0.3   # finish mode: eggs > direction

        for mv in moves:
            direction, movetype = mv

            # Base priority: EGG > TURD > MOVE
            pri = self.move_priority(mv)

            # Corner rule: penalize turds at the edge (your original idea)
            if is_corner and movetype == MoveType.TURD:
                pri = -1  # below MOVE

            # Phase-dependent adjustments
            if opp_mob > CRUSH_THRESHOLD:
                # --- Full Tal mode: kill their mobility ---
                if movetype == MoveType.TURD:
                    if turd_hot_zone and remaining_turds >= 2:
                        # In center / near opp, with ammo left: boost turd.
                        pri += 1
                    else:
                        # Outside good zones or almost out: downweight.
                        pri -= 1

            elif opp_mob > FINISH_THRESHOLD:
                # --- Crush mode: they’re cramped, but not dead ---
                if movetype == MoveType.EGG:
                    pri += 1      # eggs get a bit more important
                if movetype == MoveType.TURD:
                    pri -= 1      # conserve turds unless it's really good

            else:
                # --- Finish mode: convert position into egg lead ---
                if movetype == MoveType.EGG:
                    pri += 2      # slam eggs
                if movetype == MoveType.TURD:
                    pri -= 2      # almost never spend turds now

            dir_score = self.direction_score(board, mv)

            scored.append((pri, dir_w * dir_score, mv))

        # 4. Sort by (priority, direction_score)
        scored.sort(key=lambda t: (t[0], t[1]), reverse=True)

        return [t[2] for t in scored]


    def move_priority(self, mv) -> int:
        """
        Base move priority. Contextual adjustments happen in order_moves().
        """
        _, movetype = mv
        if movetype == MoveType.EGG:
            return 2
        if movetype == MoveType.TURD:
            return 1
        return 0
    def order_moves(self, board: board.Board, moves, blocked_dir=None):
        cur_loc = board.chicken_player.get_location()
        dim = board.game_map.MAP_SIZE
        x, y = cur_loc

        opp_pos = board.chicken_enemy.get_location()
        dist_to_opp = self.manhattan(cur_loc, opp_pos)

        # --- geometric classification -----------------------------------------
        is_corner = (x < 2 or x >= dim - 2) or (y < 2 or y >= dim - 2)
        is_center = (2 <= x <= 5) and (2 <= y <= 5)

        # "Hot zone" where offensive turds make sense:
        turd_hot_zone = is_center or (dist_to_opp <= 3)

        # Remaining turds (max 5)
        my_turds = board.turds_player
        remaining_turds = max(0, 5 - len(my_turds))

        # Opponent egg-path mobility: how free they still are to grow eggs.
        opp_mob = self.opp_egg_path_mobility(board)
        FINISH_THRESHOLD = 8.0    # basically egg-dead
        CRUSH_THRESHOLD  = 20.0   # significantly cramped

        # ---------------------------------------------------------------------
        # 1. Egg filtering (keep your original behavior)
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

        # 2. Remove moves stepping on known trapdoors (absolute)
        safe_moves = []
        for direction, movetype in moves:
            next_loc = loc_after_direction(cur_loc, direction)
            if next_loc not in self.known_traps:
                safe_moves.append((direction, movetype))
        if safe_moves:
            moves = safe_moves

        # 3. Score moves (Tal-style phases)
        scored: list[tuple[float, float, tuple[Direction, MoveType]]] = []

        # Weight of directional score by phase
        if opp_mob > CRUSH_THRESHOLD:
            dir_w = 1.0   # full Tal: directional space really matters
        elif opp_mob > FINISH_THRESHOLD:
            dir_w = 0.7   # crush mode
        else:
            dir_w = 0.3   # finish mode: eggs > direction

        for mv in moves:
            direction, movetype = mv

            # Base priority: EGG > TURD > MOVE
            pri = self.move_priority(mv)

            # Corner rule: penalize turds at the edge (your original idea)
            if is_corner and movetype == MoveType.TURD:
                pri = -1  # below MOVE

            # Phase-dependent adjustments
            if opp_mob > CRUSH_THRESHOLD:
                # --- Full Tal mode: kill their mobility ---
                if movetype == MoveType.TURD:
                    if turd_hot_zone and remaining_turds >= 2:
                        # In center / near opp, with ammo left: boost turd.
                        pri += 1
                    else:
                        # Outside good zones or almost out: downweight.
                        pri -= 1

            elif opp_mob > FINISH_THRESHOLD:
                # --- Crush mode: they’re cramped, but not dead ---
                if movetype == MoveType.EGG:
                    pri += 1      # eggs get a bit more important
                if movetype == MoveType.TURD:
                    pri -= 1      # conserve turds unless it's really good

            else:
                # --- Finish mode: convert position into egg lead ---
                if movetype == MoveType.EGG:
                    pri += 2      # slam eggs
                if movetype == MoveType.TURD:
                    pri -= 2      # almost never spend turds now

            dir_score = self.direction_score(board, mv)

            scored.append((pri, dir_w * dir_score, mv))

        # 4. Sort by (priority, direction_score)
        scored.sort(key=lambda t: (t[0], t[1]), reverse=True)

        return [t[2] for t in scored]


    def move_priority(self, mv) -> int:
        """
        Base move priority. Contextual adjustments happen in order_moves().
        """
        _, movetype = mv
        if movetype == MoveType.EGG:
            return 2
        if movetype == MoveType.TURD:
            return 1
        return 0


    def direction_score(self, cur_board: board.Board, mv) -> float:
        """
        Directional heuristic: for the chosen direction, look at the
        half-board region in that direction from our current position.

        Tal bias:
            + favor lots of unlaid squares (future space/eggs)
            - penalize opp eggs/turds (their control)
            - strongly penalize trap probability
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

        def in_region(ix: int, iy: int) -> bool:
            if direction == Direction.UP:
                return iy > my
            elif direction == Direction.DOWN:
                return iy < my
            elif direction == Direction.RIGHT:
                return ix > mx
            elif direction == Direction.LEFT:
                return ix < mx
            else:
                # STAY or weird dir: treat whole board as region
                return True

        unlaid = 0
        opp_eggs_count = 0
        opp_turds_count = 0
        trap_prob_sum = 0.0

        for ix in range(dim):
            for iy in range(dim):
                if not in_region(ix, iy):
                    continue

                pos = (ix, iy)

                if pos not in occupied:
                    unlaid += 1

                if pos in eggs_enemy:
                    opp_eggs_count += 1
                if pos in turds_enemy:
                    opp_turds_count += 1

                trap_prob_sum += self.trap_belief.prob_at(pos)

        # Tuned weights: strong positive for space, strong negative for traps.
        score = (
            1.0 * float(unlaid)
            - 0.5 * opp_eggs_count
            - 1.0 * opp_turds_count
            - 8.0 * trap_prob_sum    # high but not infinite
        )

        return score

