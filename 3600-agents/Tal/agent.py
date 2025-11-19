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
        self.trap_hard = 0.70   # hard “lava” threshold
        self.trap_block = 20 # number of turns you should completely avoid trapdoors
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
        Tal-style evaluation using only:
        - trapdoor risk at our current square
        - Voronoi-style egg space control (space_control)
        - current egg difference

        Assumes terminal positions (no legal moves) are already handled
        in alphabeta() by returning +/-INF and *not* calling evaluate().
        """

        # 1. Basic material (eggs)
        my_eggs  = cur_board.eggs_player
        opp_eggs = cur_board.eggs_enemy

        base_me   = len(my_eggs)
        base_opp  = len(opp_eggs)
        base_diff = base_me - base_opp

        # 2. Space control over *egg-eligible* squares
        my_space, opp_space = self.space_control(cur_board)
        space_gap = my_space - opp_space    # positive if we dominate future egg squares

        # 3. Local trapdoor risk at our current location
        my_pos = cur_board.chicken_player.get_location()
        trap_here = self.trap_belief.prob_at(my_pos)

        # Stepping on a trap is catastrophic: you lose space + they gain eggs.
        TRAP_LOCAL_COST = 10.0
        trap_penalty = TRAP_LOCAL_COST * trap_here

        # 4. Phase weighting based purely on opponent egg-space
        #    (how much territory they still control for future eggs)
        FINISH_THRESHOLD = 10   # they control very few egg-eligible squares
        CRUSH_THRESHOLD  = 26   # they are clearly cramped but not dead

        # a) Finish mode: opponent egg-space tiny -> convert space into eggs
        if opp_space <= FINISH_THRESHOLD:
            return (
                5.0 * base_diff      # cash in egg lead hard
                + 4.0 * space_gap    # keep them suffocated
                - 1.0 * trap_penalty
            )

        # b) Crush mode: they still have some region, but less than us
        if opp_space <= CRUSH_THRESHOLD:
            return (
                4.0 * space_gap
                + 3.0 * base_diff
                - 1.0 * trap_penalty
            )

        # c) Full Tal mode: opponent still has decent egg-space.
        #    Main job is to murder their region, eggs are secondary.
        return (
            6.0 * space_gap        # dominate reachable egg territory
            + 1.0 * base_diff      # eggs start to matter, but not primary
            - 1.2 * trap_penalty   # slightly harsher trap fear early
        )

    
    def space_control(self, cur_board: board.Board) -> tuple[int, int]:
        """
        Voronoi-style space control over *egg squares only*:
        - my_space  = # of egg-eligible squares where my shortest-path distance < opponent's
        - opp_space = # of egg-eligible squares where opponent's distance < mine

        "Egg-eligible" means:
        - square is reachable (via bfs_distances constraints)
        - square is currently empty (no egg/turd)
        - parity matches that player's egg parity
        """

        dim = cur_board.game_map.MAP_SIZE

        my_pos  = cur_board.chicken_player.get_location()
        opp_pos = cur_board.chicken_enemy.get_location()

        eggs_p = cur_board.eggs_player
        eggs_e = cur_board.eggs_enemy
        turds_p = cur_board.turds_player
        turds_e = cur_board.turds_enemy

        occupied = eggs_p | eggs_e | turds_p | turds_e

        # Determine parities:
        # (0,0) even, (0,1) odd. Using engine legality to infer roles.
        my_even  = cur_board.can_lay_egg_at_loc((0, 0))
        opp_even = cur_board.can_lay_egg_at_loc((0, 1))

        # Reuse your BFS distances with movement constraints
        my_dist  = self.bfs_distances(cur_board, my_pos,  for_me=True)
        opp_dist = self.bfs_distances(cur_board, opp_pos, for_me=False)

        my_space = 0
        opp_space = 0

        for x in range(dim):
            for y in range(dim):
                pos = (x, y)

                # Must be empty to be egg-eligible
                if pos in occupied:
                    continue

                even = ((x + y) % 2 == 0)

                d_me  = my_dist.get(pos)
                d_opp = opp_dist.get(pos)

                # If neither side can ever reach, ignore.
                if d_me is None and d_opp is None:
                    continue

                # Only opponent can reach, and parity matches opponent
                if d_me is None and d_opp is not None and even == opp_even:
                    opp_space += 1
                    continue

                # Only we can reach, and parity matches us
                if d_opp is None and d_me is not None and even == my_even:
                    my_space += 1
                    continue

                # Both can reach: compare distances, but only count for the
                # player whose parity matches this square.
                if d_me is not None and d_opp is not None:
                    if d_me < d_opp and even == my_even:
                        my_space += 1
                    elif d_opp < d_me and even == opp_even:
                        opp_space += 1
                    # ties or parity mismatch → no one gets this square

        return my_space, opp_space


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

        # Global space picture (egg-eligible Voronoi)
        my_space, opp_space = self.space_control(board)

        # Phase thresholds – match evaluate()
        FINISH_THRESHOLD = 10   # they control very few egg-eligible squares
        CRUSH_THRESHOLD  = 26   # clearly cramped but not dead

        # ---------------------------------------------------------------------
        # 1. Egg filtering (keep your original behavior)
        egg_moves = [mv for mv in moves if mv[1] == MoveType.EGG]
        if egg_moves and not is_center:
            # In non-center regions: only egg moves are kept
            moves = egg_moves
        # In center: keep all moves; eggs still sort first later

        # 1.5 Prevent immediate backtracking
        if blocked_dir is not None:
            non_backtracking: list[tuple[Direction, MoveType]] = []
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

            # Also avoid high-probability trap squares in the early/midgame
            trap_p = self.trap_belief.prob_at(next_loc)
            if trap_p >= self.trap_hard and board.turn_count <= self.trap_block:
                continue

            filtered.append((direction, movetype))

        if filtered:
            moves = filtered
        if not moves:
            return []  # fallback, alpha-beta will handle no-move cases

        # 3. Score moves (phase-based Tal logic)
        scored: list[tuple[float, float, tuple[Direction, MoveType]]] = []

        # Weight of directional score by phase (less important in finish mode)
        if opp_space > CRUSH_THRESHOLD:
            dir_w = 1.0   # full Tal: directional space matters a lot
        elif opp_space > FINISH_THRESHOLD:
            dir_w = 0.7   # crush mode
        else:
            dir_w = 0.3   # finish mode: eggs > direction bias

        # Helper: is there already a turd adjacent to a given location?
        def has_adjacent_turd(pos: tuple[int, int]) -> bool:
            px, py = pos
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nx, ny = px + dx, py + dy
                if 0 <= nx < dim and 0 <= ny < dim and (nx, ny) in my_turds:
                    return True
            return False

        for mv in moves:
            direction, movetype = mv
            next_loc = loc_after_direction(cur_loc, direction)

            # Base priority: EGG > TURD > MOVE
            pri = self.move_priority(mv)

            # Corner rule: penalize turds at the edge
            if is_corner and movetype == MoveType.TURD:
                pri -= 2   # from 1 -> -1 (below MOVE=0)

            # If we're dropping a TURD next to an existing TURD, devalue it
            if movetype == MoveType.TURD and has_adjacent_turd(next_loc):
                pri -= 1

            # Phase-dependent adjustments based on opponent space
            if opp_space > CRUSH_THRESHOLD:
                # --- Full Tal mode: expand our region, shrink theirs ---
                if movetype == MoveType.TURD:
                    if turd_hot_zone and remaining_turds >= 2:
                        pri += 1   # good place/time to use turd
                    else:
                        pri -= 1   # don't waste turds in bad zones

            elif opp_space > FINISH_THRESHOLD:
                # --- Crush mode: they’re cramped but alive ---
                if movetype == MoveType.EGG:
                    pri += 1       # eggs start to matter more
                if movetype == MoveType.TURD:
                    pri -= 1       # be more conservative with turds

            else:
                # --- Finish mode: they have very little space left ---
                if movetype == MoveType.EGG:
                    pri += 2       # slam eggs to convert
                if movetype == MoveType.TURD:
                    pri -= 2       # almost never spend turds now

            dir_score = self.direction_score(board, mv)

            scored.append((pri, dir_w * dir_score, mv))

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
        Directional heuristic aligned with evaluate():

        For the half-board region in the given direction, score:

            + (# of egg-eligible squares for us)
            - (# of egg-eligible squares for opponent)
            - weighted trap probability in that region

        Egg-eligible = empty + correct parity for that player.

        This is a cheap, 1-ply proxy for how much this move tends to
        expand our future egg space vs theirs, while avoiding traps.
        """

        direction, _ = mv
        dim = cur_board.game_map.MAP_SIZE

        my_pos = cur_board.chicken_player.get_location()
        mx, my = my_pos

        eggs_p = cur_board.eggs_player
        eggs_e = cur_board.eggs_enemy
        turds_p = cur_board.turds_player
        turds_e = cur_board.turds_enemy

        occupied = eggs_p | eggs_e | turds_p | turds_e

        # Parities: (0,0) even, (0,1) odd; we infer roles via engine legality.
        my_even  = cur_board.can_lay_egg_at_loc((0, 0))
        opp_even = cur_board.can_lay_egg_at_loc((0, 1))

        # Define the region for this direction
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
                # STAY or weird dir: treat whole board as neutral
                return True

        my_egg_space = 0
        opp_egg_space = 0
        trap_prob_sum = 0.0

        for ix in range(dim):
            for iy in range(dim):
                if not in_region(ix, iy):
                    continue

                pos = (ix, iy)

                # Trap probability always matters, even on occupied squares.
                trap_prob_sum += self.trap_belief.prob_at(pos)

                # Egg-eligible squares must be empty.
                if pos in occupied:
                    continue

                even = ((ix + iy) % 2 == 0)

                if even == my_even:
                    my_egg_space += 1
                if even == opp_even:
                    opp_egg_space += 1

        # Weights: positive for our future egg space, negative for theirs,
        # and strong negative for traps in that region.
        TRAP_DIR_WEIGHT = 6.0

        score = (
            1.0 * float(my_egg_space)
            - 1.0 * float(opp_egg_space)
            - TRAP_DIR_WEIGHT * trap_prob_sum
        )

        return score
