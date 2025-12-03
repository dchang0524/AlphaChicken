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
    
    // Time cutoff
    if (time_left() < 0.01) {
        VoronoiInfo vor = get_voronoi(state, known_traps);
        return Evaluator::evaluate(state, vor, trap_belief) + cum_risk +
               turd_weight * state.player_turds_left;
    }
    
    // Terminal
    if (GameRules::is_game_over(state)) {
        int my_eggs = state.player_eggs_laid;
        int opp_eggs = state.enemy_eggs_laid;
        return (my_eggs - opp_eggs) * 2;
    }
    
    // Leaf
    if (depth == 0) {
        VoronoiInfo vor = get_voronoi(state, known_traps);
        float base_eval = Evaluator::evaluate(state, vor, trap_belief);
        float turd_term = turd_weight * state.player_turds_left;
        float final_eval = base_eval + cum_risk + turd_term;
        return final_eval;
    }
    
    // TT lookup (left as-is; you can fix flag logic separately if you want)
    uint64_t base_key = ZobristHash::hash(state, known_traps);
    uint64_t scenario_key = (uint64_t(even_trap.x + 1) << 16) |
                            (uint64_t(even_trap.y + 1) << 8)  |
                            (uint64_t(odd_trap.x + 1) << 24)  |
                            (uint64_t(odd_trap.y + 1) << 32);
    uint64_t key = base_key ^ scenario_key;
    
    auto tt_it = transposition_table.find(key);
    bool use_entry = false;
    if (tt_it != transposition_table.end()) {
        const TTEntry& entry = tt_it->second;
        if (entry.gen == search_gen) {
            use_entry = true;
        }
    }
    
    if (use_entry && tt_it->second.depth >= depth) {
        const TTEntry& entry = tt_it->second;
        float g_stored = entry.value;
        float v_stored = g_stored + cum_risk;
        
        if (entry.flag == TT_EXACT) return v_stored;
        if (entry.flag == TT_LOWER && v_stored > alpha) alpha = v_stored;
        if (entry.flag == TT_UPPER && v_stored < beta)  beta  = v_stored;
        if (alpha >= beta) return v_stored;
    }
    
    VoronoiInfo vor = get_voronoi(state, known_traps);
    std::vector<Move> moves = GameRules::get_valid_moves(state, known_traps);
    
    if (moves.empty()) {
        int my_eggs = state.player_eggs_laid;
        int opp_eggs = state.enemy_eggs_laid;
        // No legal moves: effectively you lose 5 eggs and game ends
        return (my_eggs - opp_eggs - 5) * 2;
    }
    
    std::vector<Move> ordered_moves = Evaluator::move_order(state, moves, vor);
    
    float best_val = -INF;
    Move best_move;
    
    for (const Move& mv : ordered_moves) {
        if (time_left() < 0.01) break;
        
        Position new_pos = BitboardOps::loc_after_direction(state.chicken_player_pos, mv.dir);
        Bitboard new_bit = state.pos_to_bitboard(new_pos);
        
        // -----------------------------------
        // Trap risk: only if not visited yet
        // -----------------------------------
        float delta_risk = 0.0f;
        
        // Only consider risk if the current player has NOT visited this square on this path
        bool already_visited = (visited_squares_our & new_bit) != 0;
        
        if (!already_visited) {
            bool is_even_sq = ((new_pos.x + new_pos.y) % 2 == 0);
            bool in_potential_list = false;
            
            // If the square is in potential trap list, its risk is handled via scenarios, so skip here
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
            
            if (!in_potential_list) {
                float prob_here = trap_belief.prob_at(new_pos);
                if (prob_here > 0.0f) {
                    // Negative because stepping on a trap is bad for the current player
                    delta_risk = -trap_weight * prob_here;
                }
            }
        }
        
        // Update visited sets for THIS node (current player)
        Bitboard updated_our = visited_squares_our | new_bit;
        Bitboard updated_opp = visited_squares_opp;
        
        // Path-risk propagation: this remains your scheme
        float child_cum_risk = -(cum_risk + delta_risk);
        
        // Apply move (perspective flips: player<->enemy)
        UndoData undo = GameRules::apply_move_inplace(state, mv, even_trap, odd_trap);
        
        // After perspective flip:
        //   - child's "our" set = previous "opp" set
        //   - child's "opp" set = previous updated "our" set
        float child_val = -negamax(state,
                                   depth - 1,
                                   -beta,
                                   -alpha,
                                   trap_belief,
                                   known_traps,
                                   even_trap,
                                   odd_trap,
                                   child_cum_risk,
                                   /*visited_squares_our=*/updated_opp,
                                   /*visited_squares_opp=*/updated_our,
                                   potential_even,
                                   potential_odd,
                                   time_left);
        
        GameRules::undo_move_inplace(state, mv, undo);
        
        if (child_val > best_val) {
            best_val = child_val;
            best_move = mv;
        }
        
        if (best_val > alpha) alpha = best_val;
        if (alpha >= beta) break;
    }
    
    // Store in TT (still using g-domain; flags logic unchanged)
    float g_here = best_val - cum_risk;
    TTFlag flag;
    if (best_val <= alpha) {
        flag = TT_UPPER;
    } else if (best_val >= beta) {
        flag = TT_LOWER;
    } else {
        flag = TT_EXACT;
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
    // Promote last root best move if valid
    auto it = std::find(ordered_moves.begin(), ordered_moves.end(), last_root_best);
    if (it != ordered_moves.end()) {
        ordered_moves.erase(it);
        ordered_moves.insert(ordered_moves.begin(), last_root_best);
    }
    
    // Get potential traps
    std::vector<Position> potential_even = trap_belief.get_potential_even(0.10f);
    std::vector<Position> potential_odd = trap_belief.get_potential_odd(0.10f);
    
    std::vector<TrapScenario> scenarios = build_trap_scenarios(trap_belief, potential_even, potential_odd);
    
    // Debug: log scenario count (helps diagnose performance issues)
    if (scenarios.size() > 20) {
        std::cerr << "SCENARIO_WARNING turn:" << state.turn_count 
                  << " scenarios:" << scenarios.size() 
                  << " moves:" << ordered_moves.size() << std::endl;
    }
    
    // NOW start the time budget AFTER setup work
    // TIME BUDGET: 8 seconds per move (allocated for search only)
    double time_budget = 8.0;
    double time_start_abs = time_left(); // Absolute time remaining at start
    double time_end_abs = time_start_abs - time_budget; // Absolute time when we should stop
    
    // Iterative deepening with time budget
    Move best_move = moves[0]; // Default to first move
    float best_val = -INF;
    int target_depth = choose_max_depth(state);
    
    // Create time checker with cached time_left to reduce callback overhead
    double cached_time_left = time_left();
    int time_check_counter = 0;
    auto time_checker = [&time_left, &cached_time_left, &time_check_counter, time_end_abs]() -> double {
        // Only call time_left() every 10 checks to reduce callback overhead
        if (++time_check_counter % 10 == 0) {
            cached_time_left = time_left();
        }
        double remaining = cached_time_left - time_end_abs;
        return remaining; // Time left relative to budget
    };
    
    int actual_depth_reached = 0;
    int move_counter = 0; // Track moves for less frequent time checks
    
    for (int depth = 1; depth <= target_depth && depth <= max_depth; ++depth) {
        // Always mark that we're attempting this depth
        actual_depth_reached = depth;
        
        // Check time BEFORE starting this depth (not during)
        // Always refresh to get accurate time
        cached_time_left = time_left();
        double time_remaining = cached_time_left - time_end_abs;
        
        // Ensure we always complete at least depth 5 (very early depths are fast)
        // This prevents cutting off too early when scenario setup is slow
        if (depth > 5) {
            if (time_remaining < 0.3) {
                // If less than 0.3s remaining, stop (save time for next move)
                break;
            }
        } else if (depth > 3 && time_remaining < 0.15) {
            // For depths 4-5, be more conservative - only stop if very low on time
            break;
        }
        
        time_check_counter = 0;
        
        float current_best_val = -INF;
        Move current_best = best_move;
        float alpha = -INF;
        float beta = INF;
        
        move_counter = 0;
        for (const Move& mv : ordered_moves) {
            // Time checks...
            if (++move_counter % 3 == 0) {
                cached_time_left = time_left();
                double time_remaining = cached_time_left - time_end_abs;
                if (time_remaining < 0.05) break;
            }
            
            Position new_pos = BitboardOps::loc_after_direction(state.chicken_player_pos, mv.dir);
            Bitboard new_bit = state.pos_to_bitboard(new_pos);
            
            // Root visited sets: our player has visited only the current square so far
            Bitboard root_visited_our = BitboardOps::set_bit(0, state.chicken_player_pos.x, state.chicken_player_pos.y);
            Bitboard root_visited_opp = 0;
            
            // -----------------------------
            // Root-level trap risk (once)
            // -----------------------------
            float delta_risk_at_root = 0.0f;
            
            bool is_even_sq = ((new_pos.x + new_pos.y) % 2 == 0);
            bool in_potential_list = false;
            
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
            
            // At root we can't have revisited new_pos (path length 1), so we only gate by potential-list
            if (!in_potential_list) {
                float prob_here = trap_belief.prob_at(new_pos);
                if (prob_here > 0.0f) {
                    delta_risk_at_root = -trap_weight * prob_here;
                }
            }
            
        // Update visited sets for our root player before flipping perspective
        Bitboard updated_root_our = root_visited_our | new_bit;
        Bitboard updated_root_opp = root_visited_opp;
        
        float exp_val = 0.0f;
        
        for (const auto& scenario : scenarios) {
            Position even_trap = scenario.has_even ? scenario.even_trap : Position(-1, -1);
            Position odd_trap  = scenario.has_odd  ? scenario.odd_trap  : Position(-1, -1);
            
            float child_cum_risk = -delta_risk_at_root;
            
            // Apply move (perspective flips here)
            UndoData undo = GameRules::apply_move_inplace(state, mv, even_trap, odd_trap);
            
            // After perspective flip:
            //   - child's "our" visited = previous "opp" (still 0 at root)
            //   - child's "opp" visited = previous updated_root_our
            float val = -negamax(state,
                                depth - 1,
                                -beta,
                                -alpha,
                                trap_belief,
                                known_traps,
                                even_trap,
                                odd_trap,
                                child_cum_risk,
                                /*visited_squares_our=*/updated_root_opp,
                                /*visited_squares_opp=*/updated_root_our,
                                potential_even,
                                potential_odd,
                                time_checker);
            
            GameRules::undo_move_inplace(state, mv, undo);
            
            exp_val += scenario.weight * val;
        }
        
        if (exp_val > current_best_val) {
            current_best_val = exp_val;
            current_best = mv;
        }
        
        if (current_best_val > alpha) alpha = current_best_val;
        if (alpha >= beta) break;
    }

        
        if (current_best_val > best_val) {
            best_val = current_best_val;
            best_move = current_best;
            last_root_best = current_best;
            // actual_depth_reached already set at start of iteration
        }
    }
    
    // Log depth information to stderr for analysis
    // Format: DEPTH_LOG turn_count:depth_reached
    std::cerr << "DEPTH_LOG " << state.turn_count << ":" << actual_depth_reached << std::endl;
    
    // Log final chosen move with evaluation breakdown (once per turn)
    VoronoiInfo vor = get_voronoi(state, known_traps);
    
    // Compute evaluation breakdown (matching evaluation.cpp heuristic)
    int my_eggs = state.player_eggs_laid;
    int opp_eggs = state.enemy_eggs_laid;
    int moves_left = state.turns_left_player;
    
    int my_potential = my_eggs + std::min(0.8 * vor.my_space_egg + 0.9 * vor.my_safe_egg, moves_left/2.0);
    int opp_potential = opp_eggs + std::min(0.8 * vor.opp_space_egg + 0.9 * vor.opp_safe_egg, moves_left/2.0);
    float base_eval = my_potential - opp_potential;
    
    std::cerr << "ROOT_CHOSEN turn:" << state.turn_count
              << " depth:" << actual_depth_reached
              << " move:" << best_move.dir << "," << best_move.move_type
              << " search_eval:" << best_val
              << " eggs:" << my_eggs << "v" << opp_eggs
              << " potential:" << my_potential << "v" << opp_potential
              << " base_eval:" << base_eval << std::endl;
    
    // Follow PV for best scenario to find leaf state
    if (!scenarios.empty()) {
        // Find best scenario
        const TrapScenario* best_scenario = &scenarios[0];
        for (const auto& s : scenarios) {
            if (s.weight > best_scenario->weight) {
                best_scenario = &s;
            }
        }
        
        GameState leaf_state = state;
        Position even_trap = best_scenario->has_even ? best_scenario->even_trap : Position(-1, -1);
        Position odd_trap = best_scenario->has_odd ? best_scenario->odd_trap : Position(-1, -1);
        
        // Apply root move
        GameRules::apply_move_inplace(leaf_state, best_move, even_trap, odd_trap);
        
        int pv_depth = 0;
        // Loop down from depth-1 because we already applied the root move
        for (int d = actual_depth_reached - 1; d > 0; --d) {
            uint64_t base_key = ZobristHash::hash(leaf_state, known_traps);
            uint64_t scenario_key = (uint64_t(even_trap.x + 1) << 16) |
                                    (uint64_t(even_trap.y + 1) << 8)  |
                                    (uint64_t(odd_trap.x + 1) << 24)  |
                                    (uint64_t(odd_trap.y + 1) << 32);
            uint64_t key = base_key ^ scenario_key;
            
            auto it = transposition_table.find(key);
            if (it == transposition_table.end()) break;
            
            Move pv_move = it->second.best_move;
            GameRules::apply_move_inplace(leaf_state, pv_move, even_trap, odd_trap);
            pv_depth++;
        }
        
        // Determine eggs/pos from original player's perspective
        // If (pv_depth + 1) is even, it's opponent's turn, so state is flipped relative to root
        // If (pv_depth + 1) is odd, it's our turn, so state is normal relative to root
        // Note: pv_depth is number of moves AFTER root move.
        // Total moves = pv_depth + 1.
        
        int leaf_my_eggs, leaf_opp_eggs;
        Position leaf_my_pos, leaf_opp_pos;
        
        if ((pv_depth + 1) % 2 == 0) {
            // Even total moves -> Opponent's turn -> Perspective flipped
            leaf_my_eggs = leaf_state.enemy_eggs_laid;
            leaf_opp_eggs = leaf_state.player_eggs_laid;
            leaf_my_pos = leaf_state.chicken_enemy_pos;
            leaf_opp_pos = leaf_state.chicken_player_pos;
        } else {
            // Odd total moves -> Our turn -> Perspective normal
            leaf_my_eggs = leaf_state.player_eggs_laid;
            leaf_opp_eggs = leaf_state.enemy_eggs_laid;
            leaf_my_pos = leaf_state.chicken_player_pos;
            leaf_opp_pos = leaf_state.chicken_enemy_pos;
        }
        
        std::cerr << "LEAF_STATE depth:" << (pv_depth + 1)
                  << " eggs:" << leaf_my_eggs << "v" << leaf_opp_eggs
                  << " my_pos:" << leaf_my_pos.x << "," << leaf_my_pos.y
                  << " opp_pos:" << leaf_opp_pos.x << "," << leaf_opp_pos.y
                  << std::endl;
    }
    
    std::cerr.flush(); // Ensure it gets printed
    
    return best_move;
}

