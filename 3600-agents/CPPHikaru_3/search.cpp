#include "search.h"
#include <algorithm>
#include <cmath>
#include <iostream>

constexpr float INF = 1e8f;
const Position NO_TRAP(-1, -1);

SearchEngine::SearchEngine() {
    search_gen = 0;
    traps_fully_known = false;
    trap_weight = 150.0f;
    turd_weight = 0.0f;
    last_root_best = Move(UP, PLAIN); // Initialize to a valid move
}

VoronoiInfo SearchEngine::get_voronoi(const GameState& state, Bitboard known_traps) {
    uint64_t key = ZobristHash::hash(state, known_traps);
    auto it = voronoi_cache.find(key);
    if (it != voronoi_cache.end()) {
        return it->second;
    }
    
    VoronoiInfo vor = VoronoiAnalyzer::analyze(state, known_traps);
    voronoi_cache[key] = vor;
    return vor;
}

std::vector<TrapScenario> SearchEngine::build_trap_scenarios(
    const TrapdoorBelief& trap_belief,
    const std::vector<Position>& potential_even,
    const std::vector<Position>& potential_odd) {
    
    std::vector<TrapScenario> scenarios;
    
    std::vector<Position> evens = potential_even;
    std::vector<Position> odds = potential_odd;
    
    if (evens.empty() && odds.empty()) {
        scenarios.push_back({Position(-1, -1), Position(-1, -1), 1.0f, false, false});
        return scenarios;
    }
    
    if (evens.empty()) {
        evens.push_back(Position(-1, -1));
    }
    if (odds.empty()) {
        odds.push_back(Position(-1, -1));
    }
    
    // Calculate probabilities
    std::vector<float> even_probs;
    std::vector<float> odd_probs;
    
    if (evens.size() == 1 && evens[0] == NO_TRAP) {
        even_probs.push_back(1.0f);
    } else {
        float sum_even = 0.0f;
        for (const auto& pos : evens) {
            float p = trap_belief.prob_at(pos);
            even_probs.push_back(p);
            sum_even += p;
        }
        if (sum_even <= 0.0f) {
            float uniform = 1.0f / evens.size();
            even_probs.assign(evens.size(), uniform);
        } else {
            for (auto& p : even_probs) {
                p /= sum_even;
            }
        }
    }
    
    if (odds.size() == 1 && odds[0] == NO_TRAP) {
        odd_probs.push_back(1.0f);
    } else {
        float sum_odd = 0.0f;
        for (const auto& pos : odds) {
            float p = trap_belief.prob_at(pos);
            odd_probs.push_back(p);
            sum_odd += p;
        }
        if (sum_odd <= 0.0f) {
            float uniform = 1.0f / odds.size();
            odd_probs.assign(odds.size(), uniform);
        } else {
            for (auto& p : odd_probs) {
                p /= sum_odd;
            }
        }
    }
    
    // Build scenarios
    for (size_t i = 0; i < evens.size(); ++i) {
        for (size_t j = 0; j < odds.size(); ++j) {
            float weight = even_probs[i] * odd_probs[j];
            bool has_even = (evens[i].x != -1 || evens[i].y != -1);
            bool has_odd = (odds[j].x != -1 || odds[j].y != -1);
            scenarios.push_back({
                evens[i],
                odds[j],
                weight,
                has_even,
                has_odd
            });
        }
    }
    
    // Normalize weights
    float total = 0.0f;
    for (const auto& s : scenarios) {
        total += s.weight;
    }
    if (total > 0.0f) {
        for (auto& s : scenarios) {
            s.weight /= total;
        }
    } else {
        scenarios.clear();
        scenarios.push_back({Position(-1, -1), Position(-1, -1), 1.0f, false, false});
    }
    
    return scenarios;
}

// ADAPTIVE DEPTH FUNCTION COMMENTED OUT - USING FIXED DEPTH 14 FOR TESTING
/*
int SearchEngine::choose_max_depth(const GameState& state) {
    // TIME-BASED ADAPTIVE DEPTH
    // Try to search as deep as possible within the time budget
    // Set a high max depth cap, actual depth will be limited by time budget
    double time_left = state.player_time;
    
    // If time is very low, use minimal depth
    if (time_left < 5) return 4;
    if (time_left < 15) return 6;
    if (time_left < 40) return 7;
    
    // With 8 seconds per move budget, allow deeper searches
    // The iterative deepening will automatically stop when time runs out
    int base_depth = 25; // High cap, will be cut off by time budget
    
    // Bonus depth if not contested (less moves to consider)
    VoronoiInfo vor = get_voronoi(state, 0);
    if (vor.contested == 0) {
        base_depth = 30; // Even higher if position is quiet
    }
    
    return base_depth;
}
*/

float SearchEngine::negamax(GameState& state,
                            int depth,
                            float alpha,
                            float beta,
                            const TrapdoorBelief& trap_belief,
                            Bitboard known_traps,
                            Position even_trap,
                            Position odd_trap,
                            float cum_risk,
                            Bitboard visited_squares_our,
                            Bitboard visited_squares_opp,
                            const std::vector<Position>& potential_even,
                            const std::vector<Position>& potential_odd,
                            std::function<double()> time_left) {
    
    // CRITICAL: Save original alpha/beta for TT flag determination
    // Alpha/beta get mutated during search, but we need originals for correct TT bounds
    float alpha_orig = alpha;
    float beta_orig = beta;
    
    // Check time
    if (time_left() < 0.01) {
        VoronoiInfo vor = get_voronoi(state, known_traps);
        return Evaluator::evaluate(state, vor, trap_belief) + cum_risk +
               turd_weight * state.player_turds_left;
    }
    
    // Terminal
    if (GameRules::is_game_over(state)) {
        int my_eggs = state.player_eggs_laid;
        int opp_eggs = state.enemy_eggs_laid;
        float terminal_eval;
        if (my_eggs > opp_eggs) {
            terminal_eval = INF;
        } else if (my_eggs < opp_eggs) {
            terminal_eval = -INF;
        } else {
            terminal_eval = 0.0f;
        }
        
        // Debug logging at terminal nodes (disabled - too verbose)
        // Uncomment below if needed for debugging
        /*
        static int terminal_counter = 0;
        if (++terminal_counter % 1000 == 0) {
            std::cerr << "TERMINAL_EVAL my_eggs:" << my_eggs
                      << " opp_eggs:" << opp_eggs
                      << " result:" << terminal_eval << std::endl;
        }
        */
        
        return terminal_eval;
    }
    
    // Leaf
    if (depth == 0) {
        VoronoiInfo vor = get_voronoi(state, known_traps);
        float base_eval = Evaluator::evaluate(state, vor, trap_belief);
        float turd_term = turd_weight * state.player_turds_left;
        float final_eval = base_eval + cum_risk + turd_term;
        
        // Debug logging disabled - too verbose
        // Uncomment below and set to very high number (50000+) if needed for debugging
        /*
        static int leaf_counter = 0;
        if (++leaf_counter % 50000 == 0) {
            std::cerr << "LEAF_EVAL final_eval:" << final_eval << std::endl;
        }
        */
        
        return final_eval;
    }
    
    // TT lookup
    uint64_t base_key = ZobristHash::hash(state, known_traps);
    uint64_t scenario_key = (uint64_t(even_trap.x + 1) << 16) | (uint64_t(even_trap.y + 1) << 8) |
                            (uint64_t(odd_trap.x + 1) << 24) | (uint64_t(odd_trap.y + 1) << 32);
    uint64_t key = base_key ^ scenario_key;
    
    auto tt_it = transposition_table.find(key);
    bool use_entry = false;
    if (tt_it != transposition_table.end()) {
        const TTEntry& entry = tt_it->second;
        if (traps_fully_known || entry.gen == search_gen) {
            use_entry = true;
        }
    }
    
    if (use_entry && tt_it->second.depth >= depth) {
        const TTEntry& entry = tt_it->second;
        float g_stored = entry.value;
        float v_stored = g_stored + cum_risk;
        
        if (entry.flag == TT_EXACT) return v_stored;
        if (entry.flag == TT_LOWER && v_stored > alpha) alpha = v_stored;
        if (entry.flag == TT_UPPER && v_stored < beta) beta = v_stored;
        if (alpha >= beta) return v_stored;
    }
    
    VoronoiInfo vor = get_voronoi(state, known_traps);
    std::vector<Move> moves = GameRules::get_valid_moves(state, known_traps);
    
    if (moves.empty()) {
        return -INF;
    }
    
    std::vector<Move> ordered_moves = Evaluator::move_order(state, moves, vor);
    
    float best_val = -INF;
    Move best_move;
    
    for (const Move& mv : ordered_moves) {
        // Time check in negamax uses time_left callback
        // We keep this check for early exit
        if (time_left() < 0.01) break;
        
        // Risk calculation - track visited squares separately for each player
        // Before move: if !is_as_turn, it's our turn; if is_as_turn, it's opponent's turn
        // After move: perspective switches, so is_as_turn flips
        Position new_pos = BitboardOps::loc_after_direction(state.chicken_player_pos, mv.dir);
        
        // TEMPORARILY DISABLE VISITED SQUARES TRACKING TO MATCH PYTHON
        // Python doesn't track visited squares, so it counts risk every time
        // This matches Python's behavior exactly
        float delta_risk = 0.0f;
        
        // Only add trapdoor risk if square is NOT in potential trap lists (matches Python logic)
        {
            bool is_even_sq = ((new_pos.x + new_pos.y) % 2 == 0);
            bool in_potential_list = false;
            
            // Check if this square is in the potential trap lists
            if (is_even_sq) {
                for (const auto& pos : potential_even) {
                    if (pos.x == new_pos.x && pos.y == new_pos.y) {
                        in_potential_list = true;
                        break;
                    }
                }
            } else {
                for (const auto& pos : potential_odd) {
                    if (pos.x == new_pos.x && pos.y == new_pos.y) {
                        in_potential_list = true;
                        break;
                    }
                }
            }
            
            // Only count risk if square is NOT in potential list (matches Python)
            if (!in_potential_list) {
                float prob_here = trap_belief.prob_at(new_pos);
                if (prob_here > 0.0f) {
                    delta_risk = -trap_weight * prob_here;
                }
            }
        }
        
        float child_cum_risk = -(cum_risk + delta_risk);
        
        // TEMPORARILY DISABLE: Don't update visited sets (match Python behavior)
        Bitboard child_visited_our = visited_squares_our;
        Bitboard child_visited_opp = visited_squares_opp;
        
        // Apply move (this will switch perspective)
        UndoData undo = GameRules::apply_move_inplace(state, mv, even_trap, odd_trap);
        
        // Recurse with updated visited_squares (perspective has now switched)
        float child_val = -negamax(state, depth - 1, -beta, -alpha,
                                   trap_belief, known_traps,
                                   even_trap, odd_trap, child_cum_risk, 
                                   child_visited_our, child_visited_opp,
                                   potential_even, potential_odd, time_left);
        
        // Undo move (visited_squares is automatically restored since we passed by value)
        GameRules::undo_move_inplace(state, mv, undo);
        
        if (child_val > best_val) {
            best_val = child_val;
            best_move = mv;
        }
        
        if (best_val > alpha) {
            alpha = best_val;
        }
        if (alpha >= beta) {
            break;
        }
    }
    
    // Store in TT
    // CRITICAL: Use original alpha_orig/beta_orig for TT flag, not mutated alpha/beta
    // This prevents "deeper loses to shallower" bug where incorrect bounds corrupt search
    float g_here = best_val - cum_risk;
    TTFlag flag;
    if (best_val <= alpha_orig) {
        flag = TT_UPPER;  // All values <= alpha_orig (upper bound)
    } else if (best_val >= beta_orig) {
        flag = TT_LOWER;  // All values >= beta_orig (lower bound, caused beta cutoff)
    } else {
        flag = TT_EXACT;  // Exact value between alpha_orig and beta_orig
    }
    
    transposition_table[key] = {
        g_here,
        depth,
        flag,
        best_move,
        search_gen
    };
    
    return best_val;
}

Move SearchEngine::search_root(const GameState& state_const,
                               const TrapdoorBelief& trap_belief,
                               Bitboard known_traps,
                               int max_depth,
                               std::function<double()> time_left) {
    // Work with a mutable copy
    GameState state = state_const;
    
    // Update traps_fully_known status
    int trap_count = BitboardOps::popcount(known_traps);
    traps_fully_known = (trap_count >= 2);
    
    // Do setup work FIRST (before starting time budget)
    // Update weights
    VoronoiInfo vor_root = get_voronoi(state, known_traps);
    trap_weight = Evaluator::get_trap_weight(state, vor_root);
    turd_weight = Evaluator::get_turd_weight(state);
    
    // Update search generation
    if (!traps_fully_known || state.player_turds_left > 0) {
        search_gen++;
    }
    if (state.turn_count % 5 == 0) {
        search_gen++;
    }
    
    // Get valid moves
    std::vector<Move> moves = GameRules::get_valid_moves(state, known_traps);
    if (moves.empty()) {
        return Move();
    }
    
    // Order moves
    std::vector<Move> ordered_moves = Evaluator::move_order(state, moves, vor_root);
    // Promote last root best move ONLY if it's already in top 3
    // This prevents oscillation: if the move fell out of favor, don't force it back
    auto it = std::find(ordered_moves.begin(), ordered_moves.end(), last_root_best);
    if (it != ordered_moves.end() && std::distance(ordered_moves.begin(), it) < 3) {
        ordered_moves.erase(it);
        ordered_moves.insert(ordered_moves.begin(), last_root_best);
    }
    
    // Get potential traps
    std::vector<Position> potential_even = trap_belief.get_potential_even(0.25f);
    std::vector<Position> potential_odd = trap_belief.get_potential_odd(0.25f);
    
    std::vector<TrapScenario> scenarios = build_trap_scenarios(trap_belief, potential_even, potential_odd);
    
    // Debug: log scenario count (helps diagnose performance issues)
    if (scenarios.size() > 20) {
        std::cerr << "SCENARIO_WARNING turn:" << state.turn_count 
                  << " scenarios:" << scenarios.size() 
                  << " moves:" << ordered_moves.size() << std::endl;
    }
    
    // ADAPTIVE DEEPENING COMMENTED OUT - HARDCODED DEPTH 14 FOR TESTING
    // NOW start the time budget AFTER setup work
    // TIME BUDGET: 8 seconds per move (allocated for search only)
    // double time_budget = 8.0;
    // double time_start_abs = time_left(); // Absolute time remaining at start
    // double time_end_abs = time_start_abs - time_budget; // Absolute time when we should stop
    
    // Iterative deepening with time budget
    Move best_move = moves[0]; // Default to first move
    float best_val = -INF;
    // int target_depth = choose_max_depth(state);
    const int FIXED_DEPTH = 9; // Match Python max_depth = 9
    
    // Create time checker with cached time_left to reduce callback overhead
    // double cached_time_left = time_left();
    // int time_check_counter = 0;
    // auto time_checker = [&time_left, &cached_time_left, &time_check_counter, time_end_abs]() -> double {
    //     // Only call time_left() every 10 checks to reduce callback overhead
    //     if (++time_check_counter % 10 == 0) {
    //         cached_time_left = time_left();
    //     }
    //     double remaining = cached_time_left - time_end_abs;
    //     return remaining; // Time left relative to budget
    // };
    
    // Dummy time checker for compatibility (not used with fixed depth)
    auto time_checker = []() -> double {
        return 1000.0; // Always return plenty of time
    };
    
    int actual_depth_reached = 0;
    
    // ITERATIVE DEEPENING: Search from depth 1 up to FIXED_DEPTH (not adaptive)
    // Only update best_move if depth completes successfully (don't use incomplete results)
    for (int depth = 1; depth <= FIXED_DEPTH && depth <= max_depth; ++depth) {
        // Track if this depth completed successfully
        bool depth_completed = true;
        
        // Always mark that we're attempting this depth
        actual_depth_reached = depth;
        
        // Check time BEFORE starting this depth (not during)
        // Always refresh to get accurate time
        // cached_time_left = time_left();
        // double time_remaining = cached_time_left - time_end_abs;
        
        // Ensure we always complete at least depth 5 (very early depths are fast)
        // This prevents cutting off too early when scenario setup is slow
        // if (depth > 5) {
        //     if (time_remaining < 0.3) {
        //         // If less than 0.3s remaining, stop (save time for next move)
        //         break;
        //     }
        // } else if (depth > 3 && time_remaining < 0.15) {
        //     // For depths 4-5, be more conservative - only stop if very low on time
        //     break;
        // }
        
        // time_check_counter = 0;
        
        float current_best_val = -INF;
        Move current_best = best_move;
        float alpha = -INF;
        float beta = INF;
        
        // move_counter = 0;
        bool all_moves_searched = true;
        for (const Move& mv : ordered_moves) {
            // Check time budget less frequently (only every few moves to reduce overhead)
            // if (++move_counter % 3 == 0) {
            //     // Refresh cache and check
            //     cached_time_left = time_left();
            //     double time_remaining = cached_time_left - time_end_abs;
            //     if (time_remaining < 0.05) {
            //         all_moves_searched = false; // Mark incomplete if we break due to time
            //         depth_completed = false;
            //         break; // Stop if less than 0.05s remaining
            //     }
            // }
            
            // Risk calculation - track visited squares separately for each player
            Position new_pos = BitboardOps::loc_after_direction(state.chicken_player_pos, mv.dir);
            
            // Initialize visited_squares for our player with current position (we're at root, it's our turn)
            Bitboard root_visited_our = BitboardOps::set_bit(0, state.chicken_player_pos.x, state.chicken_player_pos.y);
            Bitboard root_visited_opp = 0; // Opponent hasn't moved yet
            
            // TEMPORARILY DISABLE VISITED SQUARES TRACKING TO MATCH PYTHON
            float delta_risk_at_root = 0.0f;
            
            // Only add trapdoor risk if square is NOT in potential trap lists (to avoid double-counting with scenarios)
            {
                bool is_even_sq = ((new_pos.x + new_pos.y) % 2 == 0);
                bool in_potential_list = false;
                
                // Check if this square is in the potential trap lists
                if (is_even_sq) {
                    for (const auto& pos : potential_even) {
                        if (pos.x == new_pos.x && pos.y == new_pos.y) {
                            in_potential_list = true;
                            break;
                        }
                    }
                } else {
                    for (const auto& pos : potential_odd) {
                        if (pos.x == new_pos.x && pos.y == new_pos.y) {
                            in_potential_list = true;
                            break;
                        }
                    }
                }
                
                // Only count risk if square is NOT in potential list (matches Python logic)
                if (!in_potential_list) {
                    float prob_here = trap_belief.prob_at(new_pos);
                    if (prob_here > 0.0f) {
                        delta_risk_at_root = -trap_weight * prob_here;
                    }
                }
            }
            
            // Add new position to our visited set for child searches
            Bitboard child_visited_our = BitboardOps::set_bit(root_visited_our, new_pos.x, new_pos.y);
            Bitboard child_visited_opp = root_visited_opp; // Opponent's visited set stays empty at root level
            
            float exp_val = 0.0f;
            
            for (const auto& scenario : scenarios) {
                Position even_trap = scenario.has_even ? scenario.even_trap : Position(-1, -1);
                Position odd_trap = scenario.has_odd ? scenario.odd_trap : Position(-1, -1);
                
                float child_cum_risk = -delta_risk_at_root;
                
                // Apply move (modify state in place - this will switch perspective)
                UndoData undo = GameRules::apply_move_inplace(state, mv, even_trap, odd_trap);
                
                // Recurse with visited_squares tracking (each scenario uses same visited sets)
                // After apply_move_inplace, perspective has switched, so opponent is now "player"
                float val = -negamax(state, depth - 1, -beta, -alpha,
                                    trap_belief, known_traps,
                                    scenario.has_even ? scenario.even_trap : Position(-1, -1),
                                    scenario.has_odd ? scenario.odd_trap : Position(-1, -1),
                                    child_cum_risk, child_visited_our, child_visited_opp,
                                    potential_even, potential_odd, time_checker);
                
                // Undo move
                GameRules::undo_move_inplace(state, mv, undo);
                
                exp_val += scenario.weight * val;
            }
            
            if (exp_val > current_best_val) {
                current_best_val = exp_val;
                current_best = mv;
            }
            
            if (current_best_val > alpha) {
                alpha = current_best_val;
            }
            if (alpha >= beta) {
                break; // Alpha-beta cutoff - this is normal, depth still completes
                // Note: We break here, but all_moves_searched will be false
                // However, alpha-beta cutoffs mean we found a provably best move, so this is OK
            }
        }
        
        // Check if we completed searching all moves OR found provable best via alpha-beta
        // If we broke early due to time (not alpha-beta), don't use incomplete results
        if (!all_moves_searched && alpha < beta) {
            // We didn't finish all moves AND didn't prove a best move via alpha-beta
            // This means incomplete search (likely time cutoff) - don't use this depth's results
            depth_completed = false;
        }
        // Note: If we broke due to alpha >= beta, we have a provable best move, so depth is valid
        // If we searched all moves, depth is also valid
        
        // Only update best move if depth completed successfully
        // If we broke early due to time, keep previous best_move
        if (depth_completed && current_best_val > best_val) {
            best_val = current_best_val;
            best_move = current_best;
            last_root_best = current_best;
        }
        
        // If depth didn't complete, break and return best_move from previous depth
        if (!depth_completed) {
            break;
        }
    } // END OF ITERATIVE DEEPENING LOOP (depths 1 to FIXED_DEPTH)
    
    // Log depth information to stderr for analysis
    // Format: DEPTH_LOG turn_count:depth_reached
    std::cerr << "DEPTH_LOG " << state.turn_count << ":" << actual_depth_reached << std::endl;
    
    // Log final chosen move with evaluation breakdown (once per turn)
    VoronoiInfo vor = get_voronoi(state, known_traps);
    
    // Compute evaluation breakdown (same logic as evaluate function)
    int my_eggs = state.player_eggs_laid;
    int opp_eggs = state.enemy_eggs_laid;
    int mat_diff = my_eggs - opp_eggs;
    float space_score = vor.vor_score;
    // Match Python exactly: moves_left = MAX_TURNS - turn_count
    int moves_left = MAX_TURNS - state.turn_count;
    int total_moves = MAX_TURNS;
    float phase_mat = std::max(0.0f, std::min(1.0f, (float)moves_left / total_moves));
    
    float max_contested = 8.0f;
    float openness = 0.0f;
    if (max_contested > 0) {
        openness = std::max(0.0f, std::min(1.0f, (float)vor.contested / max_contested));
    }
    
    float W_SPACE_MIN = 5.0f;
    float W_SPACE_MAX = 30.0f;
    float W_MAT_MIN = 5.0f;
    float W_MAT_MAX = 25.0f;
    float W_FRAG = 3.0f;
    float FRONTIER_COEFF = 0.5f;
    
    float w_space = W_SPACE_MIN + (1.0f - openness) * (W_SPACE_MAX - W_SPACE_MIN);
    if (openness == 0.0f) {
        w_space = 0.0f;
    }
    float w_mat = W_MAT_MIN + (1.0f - phase_mat) * (W_MAT_MAX - W_MAT_MIN);
    
    float frag_score = std::max(0.0f, std::min(1.0f, vor.frag_score));
    float frag_term = -W_FRAG * openness * frag_score;
    
    float max_contested_dist = (float)vor.max_contested_dist;
    float frontier_dist_term = -FRONTIER_COEFF * (1.0f - frag_score) * max_contested_dist;
    
    float space_term = w_space * space_score;
    float mat_term = w_mat * mat_diff;
    float base_eval = space_term + mat_term + frag_term + frontier_dist_term;
    
    int egg_diff = my_eggs - opp_eggs;
    
    // Also evaluate the position AFTER the chosen move to see what the search thinks it leads to
    GameState state_after_move = state;
    UndoData undo_temp = GameRules::apply_move_inplace(state_after_move, best_move, Position(-1, -1), Position(-1, -1));
    VoronoiInfo vor_after = get_voronoi(state_after_move, known_traps);
    
    // IMPORTANT: After apply_move_inplace, perspective switches!
    // Evaluator::evaluate computes from the current player's perspective (which is now the opponent)
    // So we need to negate the result to get it from our original perspective
    float base_eval_after_opp_perspective = Evaluator::evaluate(state_after_move, vor_after, trap_belief);
    float base_eval_after = -base_eval_after_opp_perspective;  // Negate to get our perspective
    
    // Get egg counts (swap because perspective switched)
    int my_eggs_after = state_after_move.enemy_eggs_laid;
    int opp_eggs_after = state_after_move.player_eggs_laid;
    int mat_diff_after = my_eggs_after - opp_eggs_after;
    
    GameRules::undo_move_inplace(state_after_move, best_move, undo_temp);
    
    std::cerr << "ROOT_CHOSEN turn:" << state.turn_count
              << " depth:" << actual_depth_reached
              << " move:" << best_move.dir << "," << best_move.move_type
              << " search_eval:" << best_val
              << " eggs:" << my_eggs << "v" << opp_eggs
              << " diff:" << egg_diff
              << (egg_diff < 0 ? " [LOSING!]" : "") << std::endl;
    
    std::cerr << "  BEFORE_MOVE: eggs:" << my_eggs << "v" << opp_eggs
              << " mat_diff:" << mat_diff
              << " space:" << space_score << " w_space:" << w_space
              << " space_term:" << space_term << " mat_term:" << mat_term
              << " base_eval:" << base_eval << std::endl;
    
    std::cerr << "  AFTER_MOVE: eggs:" << my_eggs_after << "v" << opp_eggs_after
              << " mat_diff:" << mat_diff_after
              << " base_eval:" << base_eval_after << std::endl;
    std::cerr.flush(); // Ensure it gets printed
    
    return best_move;
}


