from __future__ import annotations
from typing import List, Tuple, Dict, Optional, Set
from collections.abc import Callable
import numpy as np

from game import board as board_mod
from game.enums import MoveType, Direction, loc_after_direction

from .voronoi import VoronoiInfo
from .hiddenMarkov import TrapdoorBelief
from .zobrist import init_zobrist, zobrist_hash, TTEntry
from .heuristics import move_order
# Import all 3 Numba functions
from .fast_logic import numba_bfs, numba_voronoi, numba_evaluate

INF = 10 ** 8

class PlayerAgent:
    def __init__(self, initial_board: board_mod.Board, time_left: Callable):
        dim = initial_board.game_map.MAP_SIZE
        self.dim = dim

        self.trap_belief = TrapdoorBelief(dim)
        self.TRAP_WEIGHT = 150.0
        self.TURD_WEIGHT = 0.0
        self.known_traps: set[Tuple[int, int]] = set()
        self.potential_even: list[Tuple[int, int]] = []
        self.potential_odd:  list[Tuple[int, int]] = []
        self.max_depth = 10
        
        self.tt: Dict[int, TTEntry] = {}
        self.vor_cache: Dict[int, VoronoiInfo] = {}
        self.last_root_best: Optional[Tuple] = None
        self.search_gen: int = 0
        self.traps_fully_known: bool = False

        self.visited_counts = np.zeros((dim, dim), dtype=np.int32)
        
        # --- Incremental Obstacle Arrays ---
        self.obs_me = np.zeros((dim, dim), dtype=np.int32)
        self.obs_opp = np.zeros((dim, dim), dtype=np.int32)

        init_zobrist(dim, seed=1234567)
        self.is_warmed_up = False

    def _warmup(self):
        try: 
            dummy_grid = np.zeros((self.dim, self.dim), dtype=np.int32)
            d_me = numba_bfs(0, 0, dummy_grid, self.dim)
            d_opp = numba_bfs(self.dim-1, self.dim-1, dummy_grid, self.dim)
            numba_voronoi(d_me, d_opp, 0, 0, 0, 1, self.dim)
            numba_evaluate(0,0,10,40,0,0,0,0,0,0)
        except Exception:
            pass

    def _init_obstacles(self, board):
        """Rebuilds the obstacle grids from scratch (call once per turn)."""
        dim = self.dim
        self.obs_me.fill(0)
        self.obs_opp.fill(0)
        
        # Traps block both players
        for t in self.known_traps:
            self.obs_me[t] += 1
            self.obs_opp[t] += 1
            
        # Enemy pieces block ME
        for t in board.eggs_enemy:
            self.obs_me[t] += 1
        for t in board.turds_enemy:
            self.obs_me[t] += 1
            for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]:
                nx, ny = t[0]+dx, t[1]+dy
                if 0 <= nx < dim and 0 <= ny < dim:
                    self.obs_me[nx, ny] += 1
                    
        # My pieces block OPPONENT
        for t in board.eggs_player:
            self.obs_opp[t] += 1
        for t in board.turds_player:
            self.obs_opp[t] += 1
            for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]:
                nx, ny = t[0]+dx, t[1]+dy
                if 0 <= nx < dim and 0 <= ny < dim:
                    self.obs_opp[nx, ny] += 1

    def play(self, board: board_mod.Board, sensor_data: List[Tuple[bool, bool]], time_left: Callable):
        if not self.is_warmed_up:
            self._warmup()
            self.is_warmed_up = True
        engine_found = board.found_trapdoors
        new_found = engine_found - self.known_traps
        for pos in new_found:
            self.trap_belief.set_trapdoor(pos)
        self.known_traps |= engine_found

        my_pos = board.chicken_player.get_location()
        opp_pos = board.chicken_enemy.get_location()

        self.trap_belief.update(my_pos, sensor_data)
        self.trap_belief.mark_safe(my_pos)
        self.trap_belief.mark_safe(opp_pos)
        self._update_trap_knowledge_state(board)
        self._update_potential_traps(board, threshold=0.30)
        
        # Initialize Obstacles for the search
        self._init_obstacles(board)

        self.visited_counts.fill(0)
        self.visited_counts[my_pos[0], my_pos[1]] = 1 

        currV = self._get_voronoi(board)
        moves_left  = board.turns_left_player 
        total_moves = board.MAX_TURNS
        phase_mat   = max(0.0, min(1.0, moves_left / total_moves))
        
        # Weights (Local calc only for Trap weight)
        W_SPACE_MIN, W_SPACE_MAX = 5.0, 25.0
        W_MAT_MIN, W_MAT_MAX = 5.0, 25.0
        w_space = W_SPACE_MIN + phase_mat * (W_SPACE_MAX - W_SPACE_MIN)
        max_contested = 8.0
        openness = max(0.0, min(1.0, currV.contested / max_contested)) if max_contested > 0 else 0.0
        if openness == 0.0: w_space = 0.0
        w_mat = W_MAT_MIN + (1.0 - phase_mat) * (W_MAT_MAX - W_MAT_MIN)
        
        self.TRAP_WEIGHT = 4 * (w_space + w_mat)
        self.TURD_WEIGHT = 2 * phase_mat
        
        if not self.traps_fully_known or board.chicken_player.get_turds_left() > 0:
            self.search_gen += 1
        if board.turn_count % 5 == 0:
            self.search_gen += 1

        moves = board.get_valid_moves()
        if not moves: return None

        turns_remaining = board.turns_left_player
        start_time = time_left()
        if turns_remaining > 0:
            time_budget = start_time / (turns_remaining + 4.0)
        else:
            time_budget = start_time
        if start_time > 10.0: time_budget = min(time_budget, 3.5) 

        target_depth = self._choose_max_depth(board)
        best_move = None
        best_val = -INF

        for depth in range(1, target_depth + 1):
            if time_left() < 0.05: break
            val, mv = self._search_root(board, moves, depth, time_left)
            if mv is None: break
            best_val = val
            best_move = mv
            self.last_root_best = mv 

        if best_move is None: best_move = moves[0]
        return best_move

    def _get_voronoi(self, cur_board: board_mod.Board) -> VoronoiInfo:
        """
        Optimized to use persistent incremental arrays.
        No more loops over sets here!
        """
        key = zobrist_hash(cur_board, self.known_traps)
        vor = self.vor_cache.get(key)
        if vor is not None:
            return vor

        dim = self.dim
        mx, my = cur_board.chicken_player.get_location()
        ox, oy = cur_board.chicken_enemy.get_location()

        # Just call Numba with existing arrays
        dist_me = numba_bfs(mx, my, self.obs_me, dim)
        dist_opp = numba_bfs(ox, oy, self.obs_opp, dim)

        stats = numba_voronoi(
            dist_me, dist_opp, mx, my, 
            cur_board.chicken_player.even_chicken, 
            cur_board.chicken_enemy.even_chicken, 
            dim
        )
        
        (my_owned, opp_owned, contested, max_cd, min_cd, min_ed, 
         my_v, opp_v, cu, cr, cd, cl, q1, q2, q3, q4) = stats

        total_dir = cu + cr + cd + cl
        if total_dir <= 1:
            cardinal_frag = 0.0
        else:
            counts = [cu, cr, cd, cl]
            dir_count = sum(1 for c in counts if c > 0)
            dir_score = (dir_count - 1) / 3.0
            major_frac = max(counts) / total_dir
            spread_score = 1.0 - major_frac
            opp_bonus = 0.0
            if cl > 0 and cr > 0: opp_bonus += 0.5
            if cu > 0 and cd > 0: opp_bonus += 0.5
            opp_score = min(1.0, opp_bonus)
            cardinal_frag = 0.4*spread_score + 0.2*dir_score + 0.4*opp_score
            cardinal_frag = max(0.0, min(1.0, cardinal_frag))

        quad_counts = [q1, q2, q3, q4]
        q_total = sum(quad_counts)
        quad_dirs = sum(1 for c in quad_counts if c > 0)
        quad_spread = 1.0 - (max(quad_counts) / q_total) if q_total > 0 else 0.0
        quad_score = 0.5 * quad_spread + 0.5 * ((quad_dirs - 1) / 3.0)
        
        frag_score = 0.5 * cardinal_frag + 0.5 * quad_score
        if min_cd == 999: min_cd = dim * dim
        if min_ed == 999: min_ed = dim * dim

        vor = VoronoiInfo(
            my_owned, opp_owned, contested, max_cd, min_cd, min_ed,
            my_v, opp_v, cu, cr, cd, cl, frag_score
        )
        
        self.vor_cache[key] = vor
        return vor

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
        
        # --- INCREMENTAL OBSTACLE UPDATE ---
        # I am placing something. It blocks the OPPONENT.
        if move_type == MoveType.EGG:
            self.obs_opp[saved_loc] += 1
        elif move_type == MoveType.TURD:
            self.obs_opp[saved_loc] += 1
            for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]:
                nx, ny = saved_loc[0]+dx, saved_loc[1]+dy
                if 0 <= nx < self.dim and 0 <= ny < self.dim:
                    self.obs_opp[nx, ny] += 1

        # Standard Board Update
        if move_type == MoveType.EGG:
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
            
        my_chicken.loc = new_loc
        
        self.visited_counts[new_loc[0], new_loc[1]] += 1
        new_risk_added = (self.visited_counts[new_loc[0], new_loc[1]] == 1)

        triggered_trap = False
        if (even_trap and new_loc == even_trap) or (odd_trap and new_loc == odd_trap):
            triggered_trap = True
            board.chicken_enemy.increment_eggs_laid(4)
            my_chicken.reset_location()
            
        board.turns_left_player -= 1
        board.is_as_turn = not board.is_as_turn
        
        # PERSPECTIVE SWAP
        # 1. Swap board logic
        board.reverse_perspective()
        # 2. Swap Obstacle Array pointers so "obs_me" always points to active player's obstacles
        self.obs_me, self.obs_opp = self.obs_opp, self.obs_me

        return (saved_loc, saved_eggs_count, saved_turds_left, saved_turns, 
                saved_is_as_turn, added_to_set, triggered_trap, new_risk_added, new_loc)

    def _undo_move_inplace(self, board, move, undo_data):
        (saved_loc, saved_eggs, saved_turds, saved_turns, saved_is_as_turn, 
         added_to_set, triggered_trap, new_risk_added, new_loc) = undo_data

        # REVERT PERSPECTIVE
        board.reverse_perspective()
        self.obs_me, self.obs_opp = self.obs_opp, self.obs_me
        
        # REVERT OBSTACLES
        if move[1] == MoveType.EGG:
            self.obs_opp[saved_loc] -= 1
        elif move[1] == MoveType.TURD:
            self.obs_opp[saved_loc] -= 1
            for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]:
                nx, ny = saved_loc[0]+dx, saved_loc[1]+dy
                if 0 <= nx < self.dim and 0 <= ny < self.dim:
                    self.obs_opp[nx, ny] -= 1
        
        # REVERT BOARD
        self.visited_counts[new_loc[0], new_loc[1]] -= 1
        
        board.turns_left_player = saved_turns
        board.is_as_turn = saved_is_as_turn
        
        my_chicken = board.chicken_player
        my_chicken.loc = saved_loc
        my_chicken.eggs_laid = saved_eggs
        my_chicken.turds_left = saved_turds
        
        if triggered_trap:
             board.chicken_enemy.eggs_laid -= 4

        if added_to_set:
            if move[1] == MoveType.EGG:
                board.eggs_player.remove(saved_loc)
            elif move[1] == MoveType.TURD:
                board.turds_player.remove(saved_loc)

    def negamax(self, cur_board, depth, alpha, beta, time_left, even_trap, odd_trap, cum_risk):
        if time_left() < 0.01:
             vor = self._get_voronoi(cur_board)
             val = numba_evaluate(
                 float(cur_board.chicken_player.eggs_laid),
                 float(cur_board.chicken_enemy.eggs_laid),
                 float(cur_board.turns_left_player),
                 float(cur_board.MAX_TURNS),
                 float(vor.vor_score),
                 float(vor.contested),
                 float(vor.frag_score),
                 float(vor.max_contested_dist),
                 float(vor.min_egg_dist),
                 float(cur_board.chicken_player.get_turds_left())
             )
             return val + cum_risk

        if cur_board.is_game_over():
            my_eggs  = cur_board.chicken_player.eggs_laid
            opp_eggs = cur_board.chicken_enemy.eggs_laid
            if my_eggs > opp_eggs: return INF
            elif my_eggs < opp_eggs: return -INF
            else: return 0.0

        if depth == 0:
             vor = self._get_voronoi(cur_board)
             val = numba_evaluate(
                 float(cur_board.chicken_player.eggs_laid),
                 float(cur_board.chicken_enemy.eggs_laid),
                 float(cur_board.turns_left_player),
                 float(cur_board.MAX_TURNS),
                 float(vor.vor_score),
                 float(vor.contested),
                 float(vor.frag_score),
                 float(vor.max_contested_dist),
                 float(vor.min_egg_dist),
                 float(cur_board.chicken_player.get_turds_left())
             )
             return val + cum_risk

        base_key = zobrist_hash(cur_board, self.known_traps)
        scenario_key = hash((even_trap, odd_trap))
        key = base_key ^ scenario_key
        entry = self.tt.get(key)
        use_entry = (entry is not None) and (self.traps_fully_known or entry.gen == self.search_gen)
        if use_entry and entry.depth >= depth:
            v_stored = entry.value + cum_risk
            if entry.flag == "EXACT": return v_stored
            elif entry.flag == "LOWER": alpha = max(alpha, v_stored)
            elif entry.flag == "UPPER": beta = min(beta, v_stored)
            if alpha >= beta: return v_stored

        vor = self._get_voronoi(cur_board)
        moves = cur_board.get_valid_moves()
        if not moves: return -INF

        ordered_moves = move_order(cur_board, moves, vor)
        best_val = -INF
        best_move = None

        for mv in ordered_moves:
            if time_left() < 0.01: break
            
            direction = mv[0]
            px, py = loc_after_direction(cur_board.chicken_player.get_location(), direction)
            
            undo_data = self._apply_move_inplace(cur_board, mv, even_trap, odd_trap)
            new_risk_added = undo_data[-2]
            
            delta_risk = 0.0
            if new_risk_added:
                prob_here = self.trap_belief.prob_at((px, py))
                if prob_here > 0:
                    is_even_sq = ((px + py) % 2 == 0)
                    if (is_even_sq and (px, py) not in self.potential_even) or \
                       (not is_even_sq and (px, py) not in self.potential_odd):
                        delta_risk = -self.TRAP_WEIGHT * prob_here

            child_cum_risk = -(cum_risk + delta_risk)

            child_val = -self.negamax(
                cur_board, depth - 1, -beta, -alpha, time_left, 
                even_trap, odd_trap, cum_risk=child_cum_risk
            )
            
            self._undo_move_inplace(cur_board, mv, undo_data)

            if child_val > best_val:
                best_val = child_val
                best_move = mv
            alpha = max(alpha, best_val)
            if alpha >= beta: break

        g_here = best_val - cum_risk
        if best_move is not None:
            flag = "EXACT"
            if best_val <= alpha: flag = "UPPER"
            elif best_val >= beta: flag = "LOWER"
            self.tt[key] = TTEntry(g_here, depth, flag, best_move, self.search_gen)

        return best_val

    def _search_root(self, board, moves, depth, time_left):
        alpha, beta = -INF, INF
        best_move = None
        best_val = -INF

        vor_root = self._get_voronoi(board)
        ordered_moves = move_order(board, moves, vor_root)
        if self.last_root_best and self.last_root_best in ordered_moves:
            ordered_moves.remove(self.last_root_best)
            ordered_moves.insert(0, self.last_root_best)

        scenarios = self._build_trap_scenarios()

        for mv in ordered_moves:
            if time_left() < 0.02: break
            exp_val = 0.0

            for (even_trap, odd_trap, weight) in scenarios:
                direction = mv[0]
                px, py = loc_after_direction(board.chicken_player.get_location(), direction)
                
                undo_data = self._apply_move_inplace(board, mv, even_trap, odd_trap)
                new_risk_added = undo_data[-2]

                delta_risk = 0.0
                if new_risk_added:
                    prob_here = self.trap_belief.prob_at((px, py))
                    if prob_here > 0:
                        is_even_sq = ((px + py) % 2 == 0)
                        if (is_even_sq and (px, py) not in self.potential_even) or \
                           (not is_even_sq and (px, py) not in self.potential_odd):
                            delta_risk = -self.TRAP_WEIGHT * prob_here
                
                child_cum_risk = -(0.0 + delta_risk)

                val = -self.negamax(board, depth - 1, -beta, -alpha, time_left, even_trap, odd_trap, cum_risk=child_cum_risk)
                self._undo_move_inplace(board, mv, undo_data)
                exp_val += weight * val

            if exp_val > best_val:
                best_val = exp_val
                best_move = mv
            alpha = max(alpha, best_val)
            if alpha >= beta: break

        return best_val, best_move
    
    # ... [Keep helper methods _update_trap_knowledge_state, _update_potential_traps, _build_trap_scenarios, _choose_max_depth same as before] ...
    def _update_trap_knowledge_state(self, board):
        prev = self.traps_fully_known
        self.traps_fully_known = (len(self.known_traps) >= 2)
        if self.traps_fully_known and not prev: self.search_gen += 1
    
    def _update_potential_traps(self, board, threshold=0.30):
        dim = board.game_map.MAP_SIZE
        self.potential_even = []
        self.potential_odd  = []
        for x in range(dim):
            for y in range(dim):
                p = self.trap_belief.prob_at((x, y))
                if p < threshold: continue
                if ((x + y) & 1) == 0: self.potential_even.append((x, y))
                else: self.potential_odd.append((x, y))
    
    def _build_trap_scenarios(self):
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
                scenarios.append((e, o, pe * po))
        total_w = sum(w for _, _, w in scenarios)
        if total_w > 0.0: return [(e, o, w / total_w) for (e, o, w) in scenarios]
        else: return [(None, None, 1.0)]
    
    def _choose_max_depth(self, board):
        t = board.player_time
        if t < 5: base = 4
        elif t < 15: base = self.max_depth - 4
        elif t < 40: base = self.max_depth - 2
        elif t < 180: base = self.max_depth - 1
        else: base = self.max_depth
        vor = self._get_voronoi(board)
        if vor.contested == 0: base = min(self.max_depth, base + 4)
        return min(base, self.max_depth)