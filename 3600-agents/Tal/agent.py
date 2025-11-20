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

        #anti repetition
        self.prev_pos: Tuple[int, int] | None = None
        self.visited = [[False for _ in range(8)] for _ in range(8)]

        # Hyperparameters:
        self.max_depth = 10     # typical; drop to 2 if time is low
        self.trap_hard = 0.50   # hard “lava” threshold
        self.trap_block = 15 # number of turns you should completely avoid trapdoors
        self.trap_weight = 50 # soft risk penalty scale
        self.look_radius = 3 #for heauristic evalutions
        self.centralize_turn_cap = 4   # you can tune this


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

        moves = board.get_valid_moves()


        # # --- CENTRALIZATION PHASE ---
        # # If we are not yet in the center box, and it's still early in the game,
        # # just play a centralizing move instead of full alpha-beta.
        # if (not self.is_centralized(board)) and (board.turn_count <= self.centralize_turn_cap):
        #     central_move = self.choose_centralize_move(board, moves)
        #     self.prev_pos = my_pos
        #     return central_move

        #Alpha-Beta Search for best move
        depth = self.choose_depth(board.player_time, board.turn_count)
        if not moves:
            return (Direction.STAY, MoveType.PLAIN)

        best_move = None
        best_val = -INF

        ordered = self.order_moves(board, moves, blocked_dir=None)
        # Root-level anti-backtracking using prev_pos
        # if self.prev_pos is not None:
        #     cur_loc = my_pos
        #     filtered = []
        #     for direction, movetype in ordered:
        #         next_loc = loc_after_direction(cur_loc, direction)
        #         if next_loc != self.prev_pos:
        #             filtered.append((direction, movetype))
        #     if filtered:
        #         ordered = filtered

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

        # Hard time cutoff
        if time_left() < 0.05:
            return self.evaluate(board)

        if depth == 0:
            return self.evaluate(board)

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
                                time_left,
                                blocked_dir=None)

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


    def evaluate(self, cur_board: board.Board) -> float:

        """
        3-phase evaluation:

        Phase 1 (early, open game):
            - Primary: minimize opponent's egg Voronoi (opp_voronoi)
            - Secondary: small weight on egg difference

        Phase 2 (midgame / conversion):
            - Mix: opp_voronoi and egg difference both matter

        Phase 3 (late / finish):
            - Primary: egg difference (greedy)
            - Secondary: small leftover Voronoi term

        Phase selection depends on BOTH:
            - how many egg-eligible squares the opponent controls (opp_voronoi)
            - how many turns are left in the game.

        Trapdoors are NOT scored here; they are handled by move ordering.
        """

        dim = cur_board.game_map.MAP_SIZE

        # --- Basic material ---
        my_eggs  = cur_board.eggs_player
        opp_eggs = cur_board.eggs_enemy
        my_turds = cur_board.turds_player
        opp_turds = cur_board.turds_enemy

        base_me   = len(my_eggs)
        base_opp  = len(opp_eggs)
        base_diff = base_me - base_opp

        # --- Positions ---
        my_pos  = cur_board.chicken_player.get_location()
        opp_pos = cur_board.chicken_enemy.get_location()

        # --- Distances for both players (movement constraints respected) ---
        my_dist  = self.bfs_distances(cur_board, my_pos,  for_me=True)
        opp_dist = self.bfs_distances(cur_board, opp_pos, for_me=False)

        eggs_p = my_eggs
        eggs_e = opp_eggs
        turds_p = my_turds
        turds_e = opp_turds
        occupied = eggs_p | eggs_e | turds_p | turds_e

        # Parity: we infer from engine legality.
        # (0,0) even, (0,1) odd. Opponent is the one who can lay on (0,1).
        my_even  = cur_board.can_lay_egg_at_loc((0, 0))
        opp_even = cur_board.can_lay_egg_at_loc((0, 1))

        # --- Opponent Voronoi over egg-eligible squares ---
        opp_voronoi = 0

        for x in range(dim):
            for y in range(dim):
                pos = (x, y)

                if pos in occupied:
                    continue

                even = ((x + y) % 2 == 0)

                # Only care about squares where opponent could lay an egg
                if even != opp_even:
                    continue

                d_me  = my_dist.get(pos)
                d_opp = opp_dist.get(pos)

                # Can't reach -> irrelevant
                if d_opp is None:
                    continue

                # Opponent controls this Voronoi cell if they are strictly closer,
                # or we can't reach at all.
                if d_me is None or d_opp < d_me:
                    opp_voronoi += 1

        # --- Wasted turd penalty (kept as a small sanity term) ---
        # WASTED_TURD_COST = 1.0
        # wasted_turds = 0

        # for tx, ty in my_turds:
        #     good_adj = False
        #     for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        #         nx, ny = tx + dx, ty + dy
        #         if not (0 <= nx < dim and 0 <= ny < dim):
        #             continue
        #         pos = (nx, ny)
        #         d_me  = my_dist.get(pos)
        #         d_opp = opp_dist.get(pos)

        #         # contested or opp-closer region -> turd is doing something
        #         if d_opp is not None:
        #             if d_me is None or d_opp <= d_me:
        #                 good_adj = True
        #                 break

        #     if not good_adj:
        #         wasted_turds += 1

        # turd_penalty = WASTED_TURD_COST * wasted_turds

        # --- Phase selection: depends on opp_voronoi AND turns left ---
        moves_left = cur_board.MAX_TURNS - cur_board.turn_count

        # Heuristic thresholds – tune if you want.
        # On 8x8, opponent Voronoi ~ 0..30-ish for egg-eligible squares.
        OPP_TIGHT = 10     # they are very constricted
        OPP_LOOSE = 20     # they have a big region

        ENDGAME_TURNS = 8   # few moves left → must convert
        MIDGAME_TURNS = 16  # enough time left to still care a lot about space

        # ---- Phase calculation (before return) ----
        # Determine which phase we are in (string label for logging)
        # if moves_left <= ENDGAME_TURNS or opp_voronoi <= OPP_TIGHT:
        #     phase = "PHASE_3_FINISH"
        #     score = 7.0 * base_diff - 1.0 * opp_voronoi
        # elif moves_left >= MIDGAME_TURNS and opp_voronoi >= OPP_LOOSE:
        #     phase = "PHASE_1_TAL"
        #     score = -5.0 * opp_voronoi + 1.0 * base_diff
        # else:
        #     phase = "PHASE_2_CONVERT"
        #     score = -3.0 * opp_voronoi + 3.0 * base_diff

        # # Log the board + evaluation
        # self.debug_log(cur_board, score, phase)

        # Phase 3: finish mode (eggs prioritized)
        if moves_left <= ENDGAME_TURNS or opp_voronoi <= OPP_TIGHT:
            # Opp is already strangled OR not many moves left:
            # prioritize egg difference hard.
            return (
                10.0 * base_diff       # main term: egg count
                - 1.0 * opp_voronoi   # small residual preference for keeping them cramped
            )

        # Phase 1: full positional Tal mode (opp still quite free, plenty of time)
        if moves_left >= MIDGAME_TURNS and opp_voronoi >= OPP_LOOSE:
            return (
                -6.0 * opp_voronoi    # primary: shrink their egg Voronoi
                + 1.0 * base_diff     # eggs matter a bit but not much
            )

        # Phase 2: mixed mode (conversion)
        # Default: neither clearly early nor clearly late, or moderately cramped opp.
        return (
            -3.0 * opp_voronoi       # still care about killing their space
            + 3.0 * base_diff        # but eggs are now equally important
        )

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
        is_corner = (x < 1 or x >= dim - 1) or (y < 1 or y >= dim - 1)
        is_center = (2 <= x <= 5) and (2 <= y <= 5)

        # "Hot zone" where offensive turds make sense:
        turd_hot_zone = is_center or (dist_to_opp <= 3)

        # Remaining turds (max 5)
        my_turds = board.turds_player
        remaining_turds = max(0, 5 - len(my_turds))

        # --- Opponent space picture (for phase weighting) ---------------------
        my_space, opp_space = self.space_control(board)

        # Phase thresholds – roughly synced with evaluate()
        FINISH_THRESHOLD = 10   # they control very few egg-eligible squares
        CRUSH_THRESHOLD  = 20   # clearly cramped but not dead

        # --- Precompute opponent-Voronoi status of our adjacent squares ------
        eggs_p = board.eggs_player
        eggs_e = board.eggs_enemy
        turds_p = board.turds_player
        turds_e = board.turds_enemy
        occupied = eggs_p | eggs_e | turds_p | turds_e

        my_dist  = self.bfs_distances(board, cur_loc,  for_me=True)
        opp_dist = self.bfs_distances(board, opp_pos, for_me=False)
        opp_even = board.can_lay_egg_at_loc((0, 1))   # opponent egg parity

        adj_opp_voronoi = 0
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx, ny = x + dx, y + dy
            if not (0 <= nx < dim and 0 <= ny < dim):
                continue
            pos = (nx, ny)
            if pos in occupied:
                continue
            even = ((nx + ny) % 2 == 0)
            if even != opp_even:
                continue

            d_me  = my_dist.get(pos)
            d_opp = opp_dist.get(pos)
            if d_opp is None:
                continue
            if d_me is None or d_opp < d_me:
                adj_opp_voronoi += 1

        # ---------------------------------------------------------------------
        # 1. If we have any EGG move, do not allow PLAIN moves at all.
        has_egg = any(mv[1] == MoveType.EGG for mv in moves)
        if has_egg:
            non_plain = [mv for mv in moves if mv[1] != MoveType.PLAIN]
            if non_plain:
                moves = non_plain

        # # 1.5 Prevent immediate backtracking (still optional)
        # if blocked_dir is not None:
        #     non_backtracking: list[tuple[Direction, MoveType]] = []
        #     for direction, movetype in moves:
        #         if direction != blocked_dir:
        #             non_backtracking.append((direction, movetype))
        #     if non_backtracking:
        #         moves = non_backtracking

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
        if not moves:
            return []  # alpha-beta will see terminal

        # 2.5 If NONE of our adjacent squares are in opponent's Voronoi,
        #    don't play TURD moves (unless that would leave us with no moves).
        if adj_opp_voronoi == 0 and is_corner:
            non_turd = [mv for mv in moves if mv[1] != MoveType.TURD]
            if non_turd:
                moves = non_turd

        if not moves:
            return []  # safety, should basically never happen

        # 3. Score moves (phase-based, Tal logic)
        scored: list[tuple[float, float, tuple[Direction, MoveType]]] = []

        # Weight of directional score by phase (less important in finish mode)
        if opp_space > CRUSH_THRESHOLD:
            dir_w = 1.0   # full Tal: directional pressure matters a lot
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

            # If MANY adjacent squares are in opponent's Voronoi,
            # prioritize TURD over EGG locally.
            if adj_opp_voronoi >= 2:
                if movetype == MoveType.TURD:
                    pri += 2   # boost turd above egg
                elif movetype == MoveType.EGG:
                    # slight relative downweight vs turd
                    pri -= 1

            # Phase-dependent adjustments based on opponent global space
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
        _, movetype = mv
        if movetype == MoveType.EGG:
            return 2
        if movetype == MoveType.TURD:
            return 1
        return 0
    
    def direction_score(self, cur_board: board.Board, mv) -> float:
        """
        Directional heuristic for Tal-bot:
        Move TOWARD the opponent's region by rewarding directions where
        the opponent's reachable egg-eligible squares in that region shrink.

        We DO NOT count our own egg-space here — that caused runaway behavior.
        We ONLY measure opponent pressure.
        """

        direction, _ = mv
        dim = cur_board.game_map.MAP_SIZE

        my_pos = cur_board.chicken_player.get_location()
        mx, my = my_pos

        opp_pos = cur_board.chicken_enemy.get_location()

        eggs_p = cur_board.eggs_player
        eggs_e = cur_board.eggs_enemy
        turds_p = cur_board.turds_player
        turds_e = cur_board.turds_enemy

        occupied = eggs_p | eggs_e | turds_p | turds_e

        # Opponent egg parity
        opp_even = cur_board.can_lay_egg_at_loc((0, 1))

        # Opponent movement map
        opp_dist = self.bfs_distances(cur_board, opp_pos, for_me=False)

        # Region definition
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
                return True  # STAY = no directional bias

        opp_region_space = 0

        for ix in range(dim):
            for iy in range(dim):
                if not in_region(ix, iy):
                    continue

                pos = (ix, iy)

                # Skip occupied: egg or turd already there
                if pos in occupied:
                    continue

                # Egg-eligible for opponent?
                if ((ix + iy) % 2 == 0) != opp_even:
                    continue

                d_opp = opp_dist.get(pos)
                if d_opp is None:
                    continue  # opponent cannot reach → irrelevant

                opp_region_space += 1

        # Smaller opponent region = better.
        # So direction_score = -opp_region_space
        return -float(opp_region_space)
    
    def space_control(self, cur_board: board.Board) -> tuple[int, int]:
        """
        Voronoi-style space control over *egg-eligible* squares:

        Returns (my_space, opp_space) where:
          - Squares counted must:
              * be empty (no egg/turd)
              * NOT be adjacent to any turd (ours or theirs)
              * be reachable by the respective player
              * have correct parity for that player's eggs
        """

        dim = cur_board.game_map.MAP_SIZE

        my_pos  = cur_board.chicken_player.get_location()
        opp_pos = cur_board.chicken_enemy.get_location()

        eggs_p = cur_board.eggs_player
        eggs_e = cur_board.eggs_enemy
        turds_p = cur_board.turds_player
        turds_e = cur_board.turds_enemy

        occupied = eggs_p | eggs_e | turds_p | turds_e

        # Squares adjacent to ANY turd (ours or theirs)
        turd_adjacent: set[tuple[int, int]] = set()
        for (tx, ty) in turds_p | turds_e:
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nx, ny = tx + dx, ty + dy
                if 0 <= nx < dim and 0 <= ny < dim:
                    turd_adjacent.add((nx, ny))

        # Parity via engine legality
        my_even  = cur_board.can_lay_egg_at_loc((0, 0))
        opp_even = cur_board.can_lay_egg_at_loc((0, 1))

        # Distances with movement constraints
        my_dist  = self.bfs_distances(cur_board, my_pos,  for_me=True)
        opp_dist = self.bfs_distances(cur_board, opp_pos, for_me=False)

        my_space = 0
        opp_space = 0

        for x in range(dim):
            for y in range(dim):
                pos = (x, y)

                # Must be empty and not adjacent to any turd
                if pos in occupied or pos in turd_adjacent:
                    continue

                even = ((x + y) % 2 == 0)

                d_me  = my_dist.get(pos)
                d_opp = opp_dist.get(pos)

                # If neither side can ever reach, ignore.
                if d_me is None and d_opp is None:
                    continue

                # Our Voronoi cell (egg-eligible for us)
                if even == my_even and d_me is not None and (d_opp is None or d_me < d_opp):
                    my_space += 1

                # Opponent's Voronoi cell (egg-eligible for them)
                if even == opp_even and d_opp is not None and (d_me is None or d_opp < d_me):
                    opp_space += 1

        return my_space, opp_space
    
    def debug_log(self, cur_board: board.Board, score: float, phase: str):
        board_str = self.get_board_string(cur_board, self.known_traps)

        with open("tal_debug.log", "a") as f:
            f.write("\n===============================\n")
            f.write(f"Turn: {cur_board.turn_count}\n")
            f.write(f"Phase: {phase}\n")
            f.write(f"Eval Score: {score}\n\n")
            f.write(board_str)
            f.write("\n\n")


    def get_board_string(self, board: board.Board, trapdoors=None) -> str:
        if trapdoors is None:
            trapdoors = set()

        main_list = []
        chicken_a = board.chicken_player if board.is_as_turn else board.chicken_enemy
        chicken_b = board.chicken_enemy if board.is_as_turn else board.chicken_player

        if board.is_as_turn:
            a_loc = board.chicken_player.get_location()
            a_eggs = board.eggs_player
            a_turds = board.turds_player
            b_loc = board.chicken_enemy.get_location()
            b_eggs = board.eggs_enemy
            b_turds = board.turds_enemy
        else:
            a_loc = board.chicken_enemy.get_location()
            a_eggs = board.eggs_enemy
            a_turds = board.turds_enemy
            b_loc = board.chicken_player.get_location()
            b_eggs = board.eggs_player
            b_turds = board.turds_player

        dim = board.game_map.MAP_SIZE
        main_list.append("  ")
        for x in range(dim):
            main_list.append(f"{x} ")
        main_list.append("\n")

        for y in range(dim):
            main_list.append(f"{y} ")
            for x in range(dim):
                current_loc = (x, y)
                if a_loc == current_loc:
                    main_list.append("@ ")
                elif b_loc == current_loc:
                    main_list.append("% ")
                elif current_loc in a_eggs:
                    main_list.append("a ")
                elif current_loc in a_turds:
                    main_list.append("A ")
                elif current_loc in b_eggs:
                    main_list.append("b ")
                elif current_loc in b_turds:
                    main_list.append("B ")
                elif current_loc in trapdoors:
                    main_list.append("T ")
                else:
                    main_list.append("  ")
            main_list.append("\n")

        return "".join(main_list)

