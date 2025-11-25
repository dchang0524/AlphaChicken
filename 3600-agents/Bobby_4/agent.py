from __future__ import annotations

from typing import List, Tuple, Dict, Optional, Set
from collections.abc import Callable

# Game engine imports
from game import board as board_mod  # type: ignore
from game.enums import MoveType, Direction  # type: ignore

# Your modules
from .voronoi import analyze as voronoi_analyze, VoronoiInfo
from .hiddenMarkov import TrapdoorBelief
from .zobrist import init_zobrist, zobrist_hash, TTEntry
from .heuristics import evaluate, move_order

INF = 10 ** 8


class PlayerAgent:
    def __init__(self, initial_board: board_mod.Board, time_left: Callable):
        dim = initial_board.game_map.MAP_SIZE

        # Belief state over trapdoors
        self.trap_belief = TrapdoorBelief(dim)

        # Scale for path-risk penalty (per square)
        # delta_risk = -TRAP_WEIGHT * P(trap at square)
        self.TRAP_WEIGHT = 150.0

        # Deterministic trapdoors (walls for BFS / voronoi / zobrist)
        self.known_traps: set[Tuple[int, int]] = set()

        # Probabilistic candidates for expectimax
        self.potential_even: list[Tuple[int, int]] = []
        self.potential_odd:  list[Tuple[int, int]] = []

        self.max_depth = 10

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

        # 3) Update "fully known" status
        self._update_trap_knowledge_state(board)

        # 4) Extract probabilistic trap candidates
        self._update_potential_traps(board, threshold=0.25)

        # 5) New root: bump generation
        if not self.traps_fully_known:
            self.search_gen += 1
        if board.turn_count % 10 == 0:
            self.search_gen += 1

        # 6) Legal moves
        moves = board.get_valid_moves()
        if not moves:
            return None

        # ------------------------------------------------
        # ASPIRATION WINDOW SEARCH LOOP
        # ------------------------------------------------
        target_depth = self._choose_max_depth(board)
        
        # --- FIX: Initialize variable BEFORE the loop ---
        best_move = None 
        # ------------------------------------------------

        current_score = 0.0
        ASPIRATION_WINDOW = 50.0 

        for depth in range(1, target_depth + 1):
            if time_left() < 0.05:
                break
            
            # Default bounds (Full Window)
            alpha, beta = -INF, INF

            if depth > 1 and time_left() > 0.5:
                alpha = current_score - ASPIRATION_WINDOW
                beta  = current_score + ASPIRATION_WINDOW

            val, mv = self._search_root(board, moves, depth, time_left, alpha, beta)
            
            # Aspiration Fail Re-search
            if depth > 1 and (val <= alpha or val >= beta):
                if time_left() > 0.1: 
                    val, mv = self._search_root(board, moves, depth, time_left, -INF, INF)

            if mv is None:
                break

            current_score = val
            self.last_root_best = mv
            best_move = mv
            
        if best_move is None:
            best_move = moves[0]

        return best_move

    # ------------------------------------------------------------------
    # Track whether traps are fully known
    # ------------------------------------------------------------------
    def _update_trap_knowledge_state(self, board: board_mod.Board) -> None:
        """
        Decide when traps are "fully known".

        Simplest heuristic: game rules typically have one even-parity and
        one odd-parity trap. Once engine has revealed both, further belief
        updates shouldn't change evaluation.
        """
        prev = self.traps_fully_known

        # If you know exact rule, replace this with that.
        # For typical parity-based setup: 2 traps total.
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
        threshold: float = 0.25,
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

        # No info at all → single 'no trap info' scenario.
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
    # Root search with expectimax over trap configs
    # ------------------------------------------------------------------
    def _search_root(
        self,
        board: board_mod.Board,
        moves: List[Tuple],
        depth: int,
        time_left: Callable,
        alpha_override: float = -INF,  # <--- NEW PARAMETER
        beta_override: float = INF,    # <--- NEW PARAMETER
    ) -> Tuple[float, Optional[Tuple]]:
        
        # Initialize with the window passed from play()
        alpha = alpha_override
        beta = beta_override
        
        best_move: Optional[Tuple] = None
        best_val = -INF

        # Voronoi once at root for move ordering
        vor_root = self._get_voronoi(board)
        ordered_moves = self._order_moves(board, moves, vor_root)

        # Reuse last root best move (PV move) from previous iteration
        if self.last_root_best is not None and self.last_root_best in ordered_moves:
            ordered_moves.remove(self.last_root_best)
            ordered_moves.insert(0, self.last_root_best)

        scenarios = self._build_trap_scenarios()

        for mv in ordered_moves:
            if time_left() < 0.02:
                break

            exp_val = 0.0

            # Expectimax summation for root move
            for (even_trap, odd_trap, weight) in scenarios:
                child, delta_risk = self._simulate_move_with_traps(
                    board, mv, even_trap, odd_trap
                )

                # Root is MAX. Child is MIN. 
                # We pass -beta, -alpha to the child.
                child_cum_risk = -(0.0 + delta_risk)

                val = -self.negamax(
                    child,
                    depth - 1,
                    -beta,
                    -alpha,
                    time_left,
                    even_trap,
                    odd_trap,
                    cum_risk=child_cum_risk,
                )
                exp_val += weight * val

            if exp_val > best_val:
                best_val = exp_val
                best_move = mv

            # Alpha-Beta updating at the root
            if best_val > alpha:
                alpha = best_val
            if alpha >= beta:
                break

        # If we timed out completely and found nothing, return fail-low
        if best_move is None:
            return -INF, None
            
        return best_val, best_move

    # ------------------------------------------------------------------
    # Move simulation with trap effect + local risk
    # ------------------------------------------------------------------
    def _simulate_move_with_traps(
        self,
        cur_board: board_mod.Board,
        mv: Tuple,
        even_trap: Optional[Tuple[int, int]],
        odd_trap: Optional[Tuple[int, int]],
    ) -> Tuple[board_mod.Board, float]:
        """
        Returns:
            (child_board, delta_risk)

        delta_risk is the path-risk penalty to the player who just moved,
        in the parent's POV. Negamax will flip sign when switching POV.
        """
        direction, move_type = mv

        # Forecast move
        child = cur_board.forecast_move(direction, move_type)

        px, py = child.chicken_player.get_location()

        # Local path-risk contribution (penalty to mover at this step)
        prob_here = self.trap_belief.prob_at((px, py))
        delta_risk = 0  # negative = penalty
        if ((px + py) % 2 == 0 and (px, py) not in self.potential_even) or ((px + py) % 2 == 1 and (px, py) not in self.potential_odd):
            delta_risk = -self.TRAP_WEIGHT * prob_here  # negative = penalty

        # Check if player stepped on any hypothetical trap this scenario assigns.
        stepped_on_even = (even_trap is not None and (px, py) == even_trap)
        stepped_on_odd  = (odd_trap  is not None and (px, py) == odd_trap)

        if stepped_on_even or stepped_on_odd:
            # Opponent gains eggs
            child.chicken_enemy.increment_eggs_laid(4)
            # Player gets teleported back to spawn
            child.chicken_player.reset_location()

        # Switch perspective for negamax child
        child.reverse_perspective()

        return child, delta_risk

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
        # 1. Terminal & Time Checks
        if cur_board.is_game_over():
            my_eggs  = cur_board.chicken_player.eggs_laid
            opp_eggs = cur_board.chicken_enemy.eggs_laid
            if my_eggs > opp_eggs: return INF
            elif my_eggs < opp_eggs: return -INF
            else: return 0.0

        if time_left() < 0.01:
            vor = self._get_voronoi(cur_board)
            return evaluate(cur_board, vor, self.trap_belief) + cum_risk

        alpha_orig = alpha

        # 2. TT Lookup
        base_key = zobrist_hash(cur_board, self.known_traps)
        scenario_key = hash((even_trap, odd_trap))
        key = base_key ^ scenario_key

        entry = self.tt.get(key)
        # Only use TT if we are sure about traps OR the entry is from this specific search generation
        use_entry = (entry is not None) and (self.traps_fully_known or entry.gen == self.search_gen)

        if use_entry and entry.depth >= depth:
            v_stored = entry.value + cum_risk
            if entry.flag == "EXACT": return v_stored
            elif entry.flag == "LOWER": alpha = max(alpha, v_stored)
            elif entry.flag == "UPPER": beta = min(beta, v_stored)
            if alpha >= beta: return v_stored

        # 3. Voronoi Analysis (Needed for Pruning & Move Ordering)
        vor = self._get_voronoi(cur_board)
        # --- FUTILITY PRUNING ---
        # If at depth 1, and the current static eval is WAY below alpha,
        # we assume no single move can fix it. Prune immediately.
        if depth == 1:
            # Calculate static evaluation (score if we do nothing)
            static_eval = evaluate(cur_board, vor, self.trap_belief) + cum_risk
            
            # Margin: 90.0 allows for ~3.5 eggs of swing. 
            # If we are losing by more than that vs alpha, give up.
            FUTILITY_MARGIN = 90.0 
            
            if static_eval + FUTILITY_MARGIN < alpha:
                # Return static eval (fail-low)
                return static_eval
        # ------------------------

        # 4. Move Generation & Ordering
        moves = cur_board.get_valid_moves()
        
        if depth == 0 or not moves:
            val = evaluate(cur_board, vor, self.trap_belief) + cum_risk
            self.tt[key] = TTEntry(val - cum_risk, depth, "EXACT", None, self.search_gen)
            return val

        ordered_moves = self._order_moves(cur_board, moves, vor)
        
        # [CRITICAL FOR PVS] Put the TT best move first!
        if use_entry and entry.best_move in ordered_moves:
             ordered_moves.remove(entry.best_move)
             ordered_moves.insert(0, entry.best_move)

        best_val = -INF
        best_move = None
        
        # 5. Principal Variation Search Loop
        for i, mv in enumerate(ordered_moves):
            if time_left() < 0.01: 
                break

            child, delta_risk = self._simulate_move_with_traps(cur_board, mv, even_trap, odd_trap)
            new_cum_risk = -(cum_risk + delta_risk)

            if i == 0:
                # PV Node: Search the first move with the FULL window
                val = -self.negamax(child, depth - 1, -beta, -alpha, time_left, 
                                    even_trap, odd_trap, new_cum_risk)
            else:
                # Null Window Search: Search with a zero-width window centered on alpha
                # We are testing: "Is this move > alpha?"
                val = -self.negamax(child, depth - 1, -alpha - 1, -alpha, time_left, 
                                    even_trap, odd_trap, new_cum_risk)
                
                # If it failed high (val > alpha), our assumption that this move was worse was wrong.
                # We must re-search with the full window to get the exact score.
                if alpha < val < beta:
                    val = -self.negamax(child, depth - 1, -beta, -alpha, time_left, 
                                        even_trap, odd_trap, new_cum_risk)

            if val > best_val:
                best_val = val
                best_move = mv

            alpha = max(alpha, best_val)
            if alpha >= beta:
                break

        # 6. Store TT
        g_here = best_val - cum_risk
        
        flag = "EXACT"
        if best_val <= alpha_orig: flag = "UPPER"
        elif best_val >= beta: flag = "LOWER"
        
        if best_move is not None:
            self.tt[key] = TTEntry(g_here, depth, flag, best_move, self.search_gen)

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
        # Right now this is just your heuristic move ordering.
        # If you want deeper TT-based ordering, you’d extend this to
        # take (even_trap, odd_trap) and look up entry.best_move.
        return move_order(cur_board, moves, vor)

    # ------------------------------------------------------------------
    # Depth scheduling
    # ------------------------------------------------------------------
    def _choose_max_depth(self, board: board_mod.Board) -> int:
        t = board.player_time  # seconds left

        if t < 5:
            base = 4
        elif t < 15:
            base = self.max_depth - 3
        elif t < 40:
            base = self.max_depth - 2
        elif t < 180:
            base = self.max_depth - 1
        else:
            base = self.max_depth

        vor = self._get_voronoi(board)
        if vor.contested == 0:
            base = min(self.max_depth, base + 2)

        return min(base, self.max_depth)
