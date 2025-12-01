from __future__ import annotations

from typing import List, Tuple, Dict, Optional, Set
from collections.abc import Callable

# Game engine imports
from game import board as board_mod  # type: ignore
from game.enums import MoveType, Direction, loc_after_direction  # type: ignore

# Your modules
from .voronoi import analyze as voronoi_analyze, VoronoiInfo
from .hiddenMarkov import TrapdoorBelief
from .zobrist import init_zobrist, zobrist_hash, TTEntry
from .heuristics import evaluate, move_order, debug_evaluate
from .weights import HeuristicWeights

INF = 10 ** 8
OUTCOME_INF = HeuristicWeights.W_LOSS_PENALTY


class PlayerAgent:
    def __init__(self, initial_board: board_mod.Board, time_left: Callable):
        dim = initial_board.game_map.MAP_SIZE

        # Store start position for heuristics
        self.start_pos = initial_board.chicken_player.get_location()
        # Belief state over trapdoors
        self.trap_belief = TrapdoorBelief(dim)

        # Scale for path-risk penalty (per square)
        # delta_risk = -TRAP_WEIGHT * P(trap at square)
        self.TRAP_WEIGHT = HeuristicWeights.TRAP_WEIGHT

        # Deterministic trapdoors (walls for BFS / voronoi / zobrist)
        self.known_traps: set[Tuple[int, int]] = set()

        # Probabilistic candidates for expectimax
        self.potential_even: list[Tuple[int, int]] = []
        self.potential_odd:  list[Tuple[int, int]] = []

        self.max_depth = 9

        # Transposition table: STATIC value g(board) (no path-risk), plus generation
        self.tt: Dict[int, TTEntry] = {}

        # Voronoi cache can persist because it depends only on board + known_traps
        self.vor_cache: Dict[int, VoronoiInfo] = {}

        # Last root best move (for root move ordering across depths / turns)
        self.last_root_best: Optional[Tuple] = None

        # Generation counter for TT validity under changing beliefs
        self.search_gen: int = 0

        # Whether traps are fully known (so belief no longer changes eval)
        self.traps_fully_known: bool = False

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
        engine_found = board.found_trapdoors
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

        # 3) Update "fully known" status (and flush TT once at transition)
        self._update_trap_knowledge_state(board)

        # 4) Extract probabilistic trap candidates for expectimax
        self._update_potential_traps(board, threshold=0.35)
        
        # 4.1 update trapdoor risk weight based on game phase and openness
        currV = self._get_voronoi(board)
        
        # Openness
        max_contested = 8.0
        openness = 0.0
        if max_contested > 0:
            openness = max(0.0, min(1.0, currV.contested / max_contested))
        
        # Distance to start
        my_pos = board.chicken_player.get_location()
        dist_to_start = abs(my_pos[0] - self.start_pos[0]) + abs(my_pos[1] - self.start_pos[1])
        
        # Formula: constant * (4 + 8 * openness * voronoi_weight * dist_to_start / 2)
        term = 4.0 + 8.0 * openness * HeuristicWeights.W_VORONOI_EGG * dist_to_start / 2
        self.TRAP_WEIGHT = HeuristicWeights.W_TRAP_CONST * term
        
        # 5) New root: bump generation for TT if traps are still uncertain or some number of turns have passed
        if not self.traps_fully_known or board.chicken_player.get_turds_left() > 0:
            self.search_gen += 1
        if board.turn_count % 5 == 0:
            self.search_gen += 1

        # 6) Legal moves
        moves = board.get_valid_moves()
        if not moves:
            return None

        # 7) Iterative deepening with expectimax over trap configs
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
            self.last_root_best = mv  # keep PV move for next root

        if best_move is None:
            best_move = moves[0]

        return best_move

    # ------------------------------------------------------------------
    # Track whether traps are fully known
    # ------------------------------------------------------------------
    def _update_trap_knowledge_state(self, board : board_mod.Board) -> None:
        """
        Decide when traps are "fully known".
        """
        prev = self.traps_fully_known
        self.traps_fully_known = (len(self.known_traps) >= 2)

        # On first transition to "fully known", old TT entries are suspect
        # (they were computed under moving beliefs), so clear once.
        if self.traps_fully_known and not prev:
            self.search_gen += 1

    # ------------------------------------------------------------------
    # Potential trap candidates (for expectimax)
    # ------------------------------------------------------------------
    def _update_potential_traps(
        self,
        board: board_mod.Board,
        threshold: float = 0.30,
    ) -> None:
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
        key = zobrist_hash(cur_board, self.known_traps)
        vor = self.vor_cache.get(key)
        if vor is None:
            vor = voronoi_analyze(self, cur_board, self.known_traps)
            self.vor_cache[key] = vor
        return vor

    # ------------------------------------------------------------------
    # Build trap scenarios for expectimax
    # ------------------------------------------------------------------
    def _build_trap_scenarios(
        self,
    ) -> List[Tuple[Optional[Tuple[int, int]], Optional[Tuple[int, int]], float]]:
        evens = list(self.potential_even)
        odds  = list(self.potential_odd)

        # No info at all -> single 'no trap info' scenario.
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
            scenarios = [(None, None, 1.0)]

        return scenarios

    # ------------------------------------------------------------------
    # IN-PLACE MOVE APPLICATION (DO / UNDO)
    # ------------------------------------------------------------------
    def _apply_move_inplace(
        self, 
        board: board_mod.Board, 
        move: Tuple, 
        even_trap: Optional[Tuple[int, int]], 
        odd_trap: Optional[Tuple[int, int]]
    ) -> Tuple:
        """
        Applies a move directly to 'board' without copying it.
        Returns all data needed to undo the move.
        """
        direction, move_type = move

        # 1. SAVE STATE (For Undo)
        my_chicken = board.chicken_player
        saved_loc = my_chicken.get_location()
        saved_eggs_count = my_chicken.eggs_laid
        saved_turds_left = my_chicken.turds_left
        saved_turns = board.turns_left_player
        saved_is_as_turn = board.is_as_turn
        
        # 2. APPLY MOVE LOGIC
        # Logic matches board.apply_move exactly
        new_loc = loc_after_direction(saved_loc, direction)
        
        added_to_set = False
        
        if move_type == MoveType.EGG:
            # Corner Reward logic
            dim = board.game_map.MAP_SIZE
            if (saved_loc[0] == 0 or saved_loc[0] == dim - 1) and \
               (saved_loc[1] == 0 or saved_loc[1] == dim - 1):
                reward = board.game_map.CORNER_REWARD
            else:
                reward = 1
            my_chicken.eggs_laid += reward
            board.eggs_player.add(saved_loc)
            added_to_set = True
            
        elif move_type == MoveType.TURD:
            my_chicken.turds_left -= 1
            board.turds_player.add(saved_loc)
            added_to_set = True
            
        # Update location
        my_chicken.loc = new_loc
        
        # 3. HANDLE TRAPS (From agent simulation logic)
        triggered_trap = False
        if (even_trap and new_loc == even_trap) or (odd_trap and new_loc == odd_trap):
            triggered_trap = True
            board.chicken_enemy.eggs_laid += 4 # Opponent bonus
            my_chicken.loc = my_chicken.spawn  # Reset to spawn
            
        # 4. UPDATE GLOBAL STATE (From board.end_turn)
        board.turns_left_player -= 1
        board.is_as_turn = not board.is_as_turn
        
        # 5. REVERSE PERSPECTIVE (Crucial for Negamax!)
        board.reverse_perspective()

        return (saved_loc, saved_eggs_count, saved_turds_left, saved_turns, saved_is_as_turn, added_to_set, triggered_trap)

    def _undo_move_inplace(self, board: board_mod.Board, move: Tuple, undo_data: Tuple) -> None:
        """
        Reverses _apply_move_inplace exactly.
        """
        direction, move_type = move
        (saved_loc, saved_eggs, saved_turds, saved_turns, saved_is_as_turn, added_to_set, triggered_trap) = undo_data

        # 1. REVERSE PERSPECTIVE BACK
        board.reverse_perspective()

        # 2. RESTORE GLOBAL STATE
        board.turns_left_player = saved_turns
        board.is_as_turn = saved_is_as_turn
        
        # 3. RESTORE CHICKEN STATE
        my_chicken = board.chicken_player
        my_chicken.loc = saved_loc
        my_chicken.eggs_laid = saved_eggs
        my_chicken.turds_left = saved_turds
        
        # 4. UNDO TRAP BONUS
        if triggered_trap:
             board.chicken_enemy.eggs_laid -= 4

        # 5. REMOVE FROM SETS
        if added_to_set:
            if move_type == MoveType.EGG:
                board.eggs_player.remove(saved_loc)
            elif move_type == MoveType.TURD:
                board.turds_player.remove(saved_loc)

    # ------------------------------------------------------------------
    # Root search with expectimax over trap configs
    # ------------------------------------------------------------------
    def _search_root(
        self,
        board: board_mod.Board,
        moves: List[Tuple],
        depth: int,
        time_left: Callable,
    ) -> Tuple[float, Optional[Tuple]]:
        alpha, beta = -INF, INF
        best_move: Optional[Tuple] = None
        best_val = -INF

        # Voronoi once at root for move ordering
        vor_root = self._get_voronoi(board)
        ordered_moves = move_order(board, moves, vor_root)

        # Reuse last root best move for move ordering across turns
        if self.last_root_best is not None and self.last_root_best in ordered_moves:
            ordered_moves.remove(self.last_root_best)
            ordered_moves.insert(0, self.last_root_best)

        scenarios = self._build_trap_scenarios()

        for mv in ordered_moves:
            if time_left() < 0.02:
                break

            exp_val = 0.0

            for (even_trap, odd_trap, weight) in scenarios:
                # --- 1. CALCULATE RISK (Pre-move) ---
                direction = mv[0]
                px, py = loc_after_direction(board.chicken_player.get_location(), direction)
                
                prob_here = self.trap_belief.prob_at((px, py))
                delta_risk = 0.0
                
                # Verify parity before applying risk penalty
                is_even_sq = ((px + py) % 2 == 0)
                # Only apply risk penalty if this square is NOT in our "potential trap" list for this scenario
                # (The scenario handles the explicit trap case via triggered_trap logic)
                # BUT, technically delta_risk is "general fear" outside the explicit scenarios.
                # Simplification: Apply risk if it matches parity logic
                if (is_even_sq and (px, py) not in self.potential_even) or \
                   (not is_even_sq and (px, py) not in self.potential_odd):
                    delta_risk = -self.TRAP_WEIGHT * prob_here

                child_cum_risk = -(0.0 + delta_risk)

                # --- 2. DO (In-Place) ---
                undo_data = self._apply_move_inplace(board, mv, even_trap, odd_trap)

                # --- 3. RECURSE ---
                # Board is already swapped/updated by apply_move_inplace
                val = -self.negamax(
                    board,
                    depth - 1,
                    -beta,
                    -alpha,
                    time_left,
                    even_trap,
                    odd_trap,
                    cum_risk=child_cum_risk,
                )
                
                # --- 4. UNDO (In-Place) ---
                self._undo_move_inplace(board, mv, undo_data)
                
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
        cum_risk: float,
    ) -> float:
        
        # --- Check for explicit Game Over (Since apply_move doesn't set winner automatically) ---
        if cur_board.turns_left_player <= 0 or cur_board.turns_left_enemy <= 0:
            my_eggs  = cur_board.chicken_player.eggs_laid
            opp_eggs = cur_board.chicken_enemy.eggs_laid
            if my_eggs > opp_eggs:
                return OUTCOME_INF + my_eggs - opp_eggs
            elif my_eggs < opp_eggs:
                return -OUTCOME_INF + my_eggs - opp_eggs
            else:
                return 0.0

        if cur_board.is_game_over(): # Fallback for other conditions
            my_eggs  = cur_board.chicken_player.eggs_laid
            opp_eggs = cur_board.chicken_enemy.eggs_laid
            if my_eggs > opp_eggs:
                return OUTCOME_INF + my_eggs - opp_eggs
            elif my_eggs < opp_eggs:
                return -OUTCOME_INF + my_eggs - opp_eggs
            else:
                return 0.0

        if time_left() < 0.01:
            vor = self._get_voronoi(cur_board)
            g = evaluate(cur_board, vor, self.trap_belief)
            return g + cum_risk

        alpha_orig = alpha

        # --- TT key ---
        base_key = zobrist_hash(cur_board, self.known_traps)
        scenario_key = hash((even_trap, odd_trap))
        key = base_key ^ scenario_key

        # --- TT probe in v-space ---
        entry = self.tt.get(key)
        use_entry = False
        if entry is not None:
            if self.traps_fully_known or entry.gen == self.search_gen:
                use_entry = True

        if use_entry and entry.depth >= depth:
            g_stored = entry.value                      # static
            v_stored = g_stored + cum_risk             # convert to current node's v-space

            if entry.flag == "EXACT":
                return v_stored
            elif entry.flag == "LOWER":
                if v_stored > alpha:
                    alpha = v_stored
            elif entry.flag == "UPPER":
                if v_stored < beta:
                    beta = v_stored

            if alpha >= beta:
                return v_stored

        # --- leaf eval ---
        vor = self._get_voronoi(cur_board)
        moves = cur_board.get_valid_moves()

        if depth == 0 or not moves:
            g_leaf = evaluate(cur_board, vor, self.trap_belief)
            v_leaf = g_leaf + cum_risk
            self.tt[key] = TTEntry(
                value=g_leaf,
                depth=depth,
                flag="EXACT",
                best_move=None,
                gen=self.search_gen,
            )
            return v_leaf

        # --- internal node search ---
        ordered_moves = self._order_moves(cur_board, moves, vor)

        best_val = -INF
        best_move = None

        for mv in ordered_moves:
            if time_left() < 0.01:
                break

            # --- 1. CALCULATE RISK (Pre-move) ---
            direction = mv[0]
            px, py = loc_after_direction(cur_board.chicken_player.get_location(), direction)
            
            prob_here = self.trap_belief.prob_at((px, py))
            delta_risk = 0.0
            
            is_even_sq = ((px + py) % 2 == 0)
            if (is_even_sq and (px, py) not in self.potential_even) or \
               (not is_even_sq and (px, py) not in self.potential_odd):
                delta_risk = -self.TRAP_WEIGHT * prob_here

            child_cum_risk = -(cum_risk + delta_risk)

            # --- 2. DO (In-Place) ---
            undo_data = self._apply_move_inplace(cur_board, mv, even_trap, odd_trap)

            # --- 3. RECURSE ---
            child_val = -self.negamax(
                cur_board,
                depth - 1,
                -beta,
                -alpha,
                time_left,
                even_trap,
                odd_trap,
                cum_risk=child_cum_risk,
            )
            
            # --- 4. UNDO (In-Place) ---
            self._undo_move_inplace(cur_board, mv, undo_data)

            if child_val > best_val:
                best_val = child_val
                best_move = mv

            if best_val > alpha:
                alpha = best_val
            if alpha >= beta:
                break

        # recover static g from v at this node
        g_here = best_val - cum_risk

        if best_move is not None:
            if best_val <= alpha_orig:
                flag = "UPPER"
            elif best_val >= beta:
                flag = "LOWER"
            else:
                flag = "EXACT"

            self.tt[key] = TTEntry(
                value=g_here,
                depth=depth,
                flag=flag,
                best_move=best_move,
                gen=self.search_gen,
            )

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
        t = board.player_time

        if t < 5:
            base = 4
        elif t < 15:
            base = self.max_depth - 3
        elif t < 40:
            base = self.max_depth - 2
        elif t < 150:
            base = self.max_depth - 1
        else:
            base = self.max_depth

        vor = self._get_voronoi(board)
        if vor.contested == 0:
            base = min(self.max_depth, base + 2)

        return min(base, self.max_depth)