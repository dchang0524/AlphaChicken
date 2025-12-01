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
from .heuristics import evaluate, move_order
from .weights import HeuristicWeights # Ensure this is imported if you use it, or define constants here

INF = 10 ** 8

class PlayerAgent:
    def __init__(self, initial_board: board_mod.Board, time_left: Callable):
        dim = initial_board.game_map.MAP_SIZE

        self.trap_belief = TrapdoorBelief(dim)

        # --- THE FIX: STATIC BALANCED WEIGHT ---
        # 350.0 creates a "rational" fear.
        # It respects the HMM probabilities without arbitrary distance penalties.
        self.TRAP_WEIGHT = 350.0 
        self.TURD_WEIGHT = 0.0

        self.known_traps: set[Tuple[int, int]] = set()
        self.potential_even: list[Tuple[int, int]] = []
        self.potential_odd:  list[Tuple[int, int]] = []

        self.max_depth = 8
        self.tt: Dict[int, TTEntry] = {}
        self.vor_cache: Dict[int, VoronoiInfo] = {}
        self.last_root_best: Optional[Tuple] = None
        self.search_gen: int = 0
        self.traps_fully_known: bool = False

        init_zobrist(dim, seed=1234567)

    def play(
        self,
        board: board_mod.Board,
        sensor_data: List[Tuple[bool, bool]],
        time_left: Callable,
    ):
        # 1. Update Beliefs
        engine_found = board.found_trapdoors
        new_found = engine_found - self.known_traps
        for pos in new_found:
            self.trap_belief.set_trapdoor(pos)
        self.known_traps |= engine_found

        my_pos  = board.chicken_player.get_location()
        opp_pos = board.chicken_enemy.get_location()

        self.trap_belief.update(my_pos, sensor_data)
        self.trap_belief.mark_safe(my_pos)
        self.trap_belief.mark_safe(opp_pos)

        self._update_trap_knowledge_state(board)
        self._update_potential_traps(board, threshold=0.40)

        # --------------------------------------------------------
        # REMOVED LEASH LOGIC HERE
        # We no longer calculate 'term' based on 'dist_to_start'.
        # We rely on the static self.TRAP_WEIGHT = 350.0 set in __init__
        # --------------------------------------------------------
        
        # 5. Search Gen Update
        if not self.traps_fully_known or board.chicken_player.get_turds_left() > 0:
            self.search_gen += 1
        if board.turn_count % 5 == 0:
            self.search_gen += 1

        # 6. Moves
        moves = board.get_valid_moves()
        if not moves:
            return None

        # 7. Search
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
            self.last_root_best = mv 

        if best_move is None:
            best_move = moves[0]

        return best_move

    # ... [Keep all other methods (_update_trap_knowledge, _update_potential_traps, etc.) EXACTLY the same] ...
    
    # Just ensure _search_root and negamax use self.TRAP_WEIGHT as defined above.
    # (The rest of your file looked correct, just strip the 'dist_to_start' block from play())

    def _update_trap_knowledge_state(self, board: board_mod.Board) -> None:
        prev = self.traps_fully_known
        self.traps_fully_known = (len(self.known_traps) >= 2)
        if self.traps_fully_known and not prev:
            self.search_gen += 1

    def _update_potential_traps(self, board: board_mod.Board, threshold: float = 0.40) -> None:
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

    def _get_voronoi(self, cur_board: board_mod.Board) -> VoronoiInfo:
        key = zobrist_hash(cur_board, self.known_traps)
        vor = self.vor_cache.get(key)
        if vor is None:
            # Note: We use the 3-argument call here compatible with your standard voronoi.py
            # If you are NOT using the Lava/BFS update, voronoi.py analyze takes (self, board, traps).
            vor = voronoi_analyze(self, cur_board, self.known_traps)
            self.vor_cache[key] = vor
        return vor

    def _build_trap_scenarios(self) -> List[Tuple[Optional[Tuple[int, int]], Optional[Tuple[int, int]], float]]:
        evens = list(self.potential_even)
        odds  = list(self.potential_odd)

        if not evens and not odds: return [(None, None, 1.0)]

        if not evens:
            evens = [None]; even_probs = [1.0]
        else:
            even_probs = [self.trap_belief.prob_at(pos) for pos in evens]
            s = sum(even_probs)
            if s <= 0.0: even_probs = [1.0/len(evens)]*len(evens)
            else: even_probs = [p/s for p in even_probs]

        if not odds:
            odds = [None]; odd_probs = [1.0]
        else:
            odd_probs = [self.trap_belief.prob_at(pos) for pos in odds]
            s = sum(odd_probs)
            if s <= 0.0: odd_probs = [1.0/len(odds)]*len(odds)
            else: odd_probs = [p/s for p in odd_probs]

        scenarios = []
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

    def _apply_move_inplace(self, board, move, even_trap, odd_trap):
        direction, move_type = move
        my_chicken = board.chicken_player
        saved_loc = my_chicken.get_location()
        saved_eggs_count = my_chicken.eggs_laid
        saved_turds_left = my_chicken.turds_left
        saved_turns = board.turns_left_player
        saved_is_as_turn = board.is_as_turn
        
        new_loc = loc_after_direction(saved_loc, direction)
        
        added_to_set = False
        if move_type == MoveType.EGG:
            dim = board.game_map.MAP_SIZE
            if (saved_loc[0] == 0 or saved_loc[0] == dim - 1) and (saved_loc[1] == 0 or saved_loc[1] == dim - 1):
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
            
        my_chicken.loc = new_loc
        
        triggered_trap = False
        if (even_trap and new_loc == even_trap) or (odd_trap and new_loc == odd_trap):
            triggered_trap = True
            board.chicken_enemy.increment_eggs_laid(4)
            my_chicken.reset_location()
            
        board.turns_left_player -= 1
        board.is_as_turn = not board.is_as_turn
        board.reverse_perspective()
        return (saved_loc, saved_eggs_count, saved_turds_left, saved_turns, saved_is_as_turn, added_to_set, triggered_trap)

    def _undo_move_inplace(self, board, move, undo_data):
        direction, move_type = move
        (saved_loc, saved_eggs, saved_turds, saved_turns, saved_is_as_turn, added_to_set, triggered_trap) = undo_data
        board.reverse_perspective()
        board.turns_left_player = saved_turns
        board.is_as_turn = saved_is_as_turn
        my_chicken = board.chicken_player
        my_chicken.loc = saved_loc
        my_chicken.eggs_laid = saved_eggs
        my_chicken.turds_left = saved_turds
        if triggered_trap:
             board.chicken_enemy.eggs_laid -= 4
        if added_to_set:
            if move_type == MoveType.EGG:
                board.eggs_player.remove(saved_loc)
            elif move_type == MoveType.TURD:
                board.turds_player.remove(saved_loc)

    def _search_root(self, board, moves, depth, time_left):
        alpha, beta = -INF, INF
        best_move = None
        best_val = -INF

        vor_root = self._get_voronoi(board)
        ordered_moves = move_order(board, moves, vor_root)

        if self.last_root_best is not None and self.last_root_best in ordered_moves:
            ordered_moves.remove(self.last_root_best)
            ordered_moves.insert(0, self.last_root_best)

        scenarios = self._build_trap_scenarios()

        for mv in ordered_moves:
            if time_left() < 0.05: return -INF, None

            exp_val = 0.0
            for (even_trap, odd_trap, weight) in scenarios:
                direction = mv[0]
                px, py = loc_after_direction(board.chicken_player.get_location(), direction)
                prob_here = self.trap_belief.prob_at((px, py))
                
                delta_risk = 0.0
                is_even_sq = ((px + py) % 2 == 0)
                if (is_even_sq and (px, py) not in self.potential_even) or \
                   (not is_even_sq and (px, py) not in self.potential_odd):
                    if prob_here > 0:
                        delta_risk = -self.TRAP_WEIGHT * prob_here

                child_cum_risk = -(0.0 + delta_risk)

                undo_data = self._apply_move_inplace(board, mv, even_trap, odd_trap)
                val = -self.negamax(board, depth - 1, -beta, -alpha, time_left, even_trap, odd_trap, child_cum_risk)
                self._undo_move_inplace(board, mv, undo_data)
                exp_val += weight * val

            if exp_val > best_val:
                best_val = exp_val
                best_move = mv
            if best_val > alpha: alpha = best_val
            if alpha >= beta: break

        if best_move is None: return -INF, None
        return best_val, best_move

    def negamax(self, cur_board, depth, alpha, beta, time_left, even_trap, odd_trap, cum_risk):
        if time_left() < 0.05:
             vor = self._get_voronoi(cur_board)
             return evaluate(cur_board, vor, self.trap_belief) + cum_risk + self.TURD_WEIGHT * cur_board.chicken_player.get_turds_left()

        if cur_board.is_game_over():
            my_eggs  = cur_board.chicken_player.eggs_laid
            opp_eggs = cur_board.chicken_enemy.eggs_laid
            if my_eggs > opp_eggs: return INF
            elif my_eggs < opp_eggs: return -INF
            else: return 0.0

        if depth == 0:
             vor = self._get_voronoi(cur_board)
             return evaluate(cur_board, vor, self.trap_belief) + cum_risk + self.TURD_WEIGHT * cur_board.chicken_player.get_turds_left()

        base_key = zobrist_hash(cur_board, self.known_traps)
        scenario_key = hash((even_trap, odd_trap))
        key = base_key ^ scenario_key

        entry = self.tt.get(key)
        use_entry = False
        if entry is not None:
            if self.traps_fully_known or entry.gen == self.search_gen:
                use_entry = True

        if use_entry and entry.depth >= depth:
            g_stored = entry.value
            v_stored = g_stored + cum_risk
            if entry.flag == "EXACT": return v_stored
            elif entry.flag == "LOWER":
                if v_stored > alpha: alpha = v_stored
            elif entry.flag == "UPPER":
                if v_stored < beta: beta = v_stored
            if alpha >= beta: return v_stored

        vor = self._get_voronoi(cur_board)
        moves = cur_board.get_valid_moves()
        if not moves: return -INF

        ordered_moves = move_order(cur_board, moves, vor)
        best_val = -INF
        best_move = None

        for mv in ordered_moves:
            if time_left() < 0.05: break

            direction = mv[0]
            px, py = loc_after_direction(cur_board.chicken_player.get_location(), direction)
            prob_here = self.trap_belief.prob_at((px, py))
            delta_risk = 0.0
            
            is_even_sq = ((px + py) % 2 == 0)
            if (is_even_sq and (px, py) not in self.potential_even) or \
               (not is_even_sq and (px, py) not in self.potential_odd):
                 if prob_here > 0:
                     delta_risk = -self.TRAP_WEIGHT * prob_here

            child_cum_risk = -(cum_risk + delta_risk)

            undo_data = self._apply_move_inplace(cur_board, mv, even_trap, odd_trap)
            child_val = -self.negamax(cur_board, depth - 1, -beta, -alpha, time_left, even_trap, odd_trap, child_cum_risk)
            self._undo_move_inplace(cur_board, mv, undo_data)

            if child_val > best_val:
                best_val = child_val
                best_move = mv
            if best_val > alpha: alpha = best_val
            if alpha >= beta: break

        g_here = best_val - cum_risk
        if best_move is not None:
            if best_val <= alpha: flag = "UPPER"
            elif best_val >= beta: flag = "LOWER"
            else: flag = "EXACT"
            self.tt[key] = TTEntry(g_here, depth, flag, best_move, self.search_gen)

        return best_val
    
    def _order_moves(self, cur_board, moves, vor):
        return move_order(cur_board, moves, vor)
    
    def _choose_max_depth(self, board):
        t = board.player_time
        if t < 5: base = 4
        elif t < 15: base = self.max_depth - 3
        elif t < 40: base = self.max_depth - 2
        elif t < 180: base = self.max_depth - 1
        else: base = self.max_depth
        vor = self._get_voronoi(board)
        if vor.contested == 0: base = min(self.max_depth, base + 2)
        return min(base, self.max_depth)