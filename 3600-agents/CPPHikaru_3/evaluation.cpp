#include "evaluation.h"
#include "game_rules.h"
#include "bfs.h"
#include <algorithm>
#include <cmath>
#include <iostream>

constexpr float INF = 1e8f;

float Evaluator::evaluate(const GameState& state, const VoronoiInfo& vor, 
                         const TrapdoorBelief& trap_belief,
                         int root_moves_left,
                         Bitboard known_traps) {
    if (GameRules::is_game_over(state)) {
        int my_eggs = state.player_eggs_laid;
        int opp_eggs = state.enemy_eggs_laid;
        if (my_eggs > opp_eggs) return INF;
        if (my_eggs < opp_eggs) return -INF;
        return 0.0f;
    }
    
    // ============================================================================
    // NEW MATERIAL HEURISTIC (CAN BE REVERTED)
    // ============================================================================
    // This heuristic counts material as:
    //   my_eggs = eggs_laid + 0.8 * (territory in my voronoi that opp can reach) 
    //           + 0.9 * (territory in my voronoi that opp can't reach)
    // 
    // To REVERT to old heuristic:
    //   1. Comment out the BFS computation and territory counting loop (lines ~27-78)
    //   2. Uncomment the OLD MATERIAL CALCULATION (lines ~80-83)
    //   3. Remove known_traps parameter from evaluate() signature
    //   4. Update all evaluate() calls to remove known_traps parameter
    // ============================================================================
    
    // Get BFS distances to determine reachability
    
    // Get BFS distances to determine reachability
    BFSResult bfs = BFS::bfs_distances_both(state, known_traps);
    
    // Compute my_eggs using new heuristic
    float my_eggs_heuristic = (float)state.player_eggs_laid;
    float opp_eggs_heuristic = (float)state.enemy_eggs_laid;
    
    // Count territory in my voronoi region
    bool my_even = (state.player_even_chicken == 0);
    bool opp_even = (state.enemy_even_chicken == 0);
    
    for (int x = 0; x < MAP_SIZE; ++x) {
        for (int y = 0; y < MAP_SIZE; ++y) {
            Position pos(x, y);
            Bitboard pos_bb = state.pos_to_bitboard(pos);
            
            // Skip squares that already have eggs on them (these are "claimed")
            if ((pos_bb & state.eggs_player) != 0 || (pos_bb & state.eggs_enemy) != 0) {
                continue;
            }
            
            int d_me = bfs.dist_me[x][y];
            int d_opp = bfs.dist_opp[x][y];
            
            bool reachable_me = (d_me >= 0);
            bool reachable_opp = (d_opp >= 0);
            
            bool is_even_sq = BitboardOps::is_even_square(pos);
            
            // Determine if square is in my voronoi region (unclaimed eggs/territory)
            if (reachable_me && (!reachable_opp || d_me <= d_opp)) {
                // Check parity (only count squares I can actually lay eggs on)
                if (is_even_sq == my_even) {
                    if (reachable_opp) {
                        // Opponent can reach this square - weight 0.8
                        my_eggs_heuristic += 0.8f;
                    } else {
                        // Opponent can't reach this square - weight 0.9
                        my_eggs_heuristic += 0.9f;
                    }
                }
            }
            
            // Symmetric for opponent (unclaimed eggs in opponent's voronoi)
            if (reachable_opp && (!reachable_me || d_opp < d_me)) {
                if (is_even_sq == opp_even) {
                    if (reachable_me) {
                        opp_eggs_heuristic += 0.8f;
                    } else {
                        opp_eggs_heuristic += 0.9f;
                    }
                }
            }
        }
    }
    
    // OLD MATERIAL CALCULATION (COMMENTED OUT FOR REVERT)
    // int my_eggs = state.player_eggs_laid;
    // int opp_eggs = state.enemy_eggs_laid;
    // int mat_diff = my_eggs - opp_eggs;
    
    // NEW: Use heuristic-based material difference
    float mat_diff = my_eggs_heuristic - opp_eggs_heuristic;
    
    float space_score = vor.vor_score;
    
    // Phase terms
    // Use root_moves_left (from root position) so weights don't change during search
    // Only change weights between actual game turns, not during search
    int moves_left = root_moves_left;
    int total_moves = MAX_TURNS;
    float phase_mat = std::max(0.0f, std::min(1.0f, (float)moves_left / total_moves));
    
    // Openness
    float max_contested = 8.0f;
    float openness = 0.0f;
    if (max_contested > 0) {
        openness = std::max(0.0f, std::min(1.0f, (float)vor.contested / max_contested));
    }
    
    // GRADUAL WEIGHT TRANSITION: Space starts high, Material starts low
    // As game progresses, Space decreases and Material increases smoothly
    // Using a smooth curve (squared) for gradual transition
    
    // Game phase: 1.0 = early game (40 moves left), 0.0 = endgame (0 moves left)
    float game_phase = phase_mat;
    float transition = 1.0f - (game_phase * game_phase); // Smooth quadratic curve
    
    // Early game weights (game_phase = 1.0, transition = 0.0)
    float EARLY_W_SPACE_MIN = 50.0f;   // High space weight early
    float EARLY_W_SPACE_MAX = 70.0f;
    float EARLY_W_MAT_MIN = 50.0f;     // Material equal to space early game
    float EARLY_W_MAT_MAX = 70.0f;
    
    // Late game weights (game_phase = 0.0, transition = 1.0)
    float LATE_W_SPACE_MIN = 10.0f;    // Lower space weight late
    float LATE_W_SPACE_MAX = 20.0f;
    float LATE_W_MAT_MIN = 60.0f;      // Much higher material weight late game
    float LATE_W_MAT_MAX = 80.0f;
    
    // Interpolate between early and late game weights
    float W_SPACE_MIN = EARLY_W_SPACE_MIN + transition * (LATE_W_SPACE_MIN - EARLY_W_SPACE_MIN);
    float W_SPACE_MAX = EARLY_W_SPACE_MAX + transition * (LATE_W_SPACE_MAX - EARLY_W_SPACE_MAX);
    float W_MAT_MIN = EARLY_W_MAT_MIN + transition * (LATE_W_MAT_MIN - EARLY_W_MAT_MIN);
    float W_MAT_MAX = EARLY_W_MAT_MAX + transition * (LATE_W_MAT_MAX - EARLY_W_MAT_MAX);
    
    // Other weights - keep fragmentation but scale it down as game progresses
    float W_FRAG = 3.0f * game_phase;  // Fragmentation less important late game
    float W_FRONTIER_DIST = 1.5f * game_phase;
    float FRONTIER_COEFF = 0.5f * game_phase;
    
    // Apply openness to space weight (space less important when board is open)
    float w_space = W_SPACE_MIN + (phase_mat) * (W_SPACE_MAX - W_SPACE_MIN);
    if (openness == 0.0f) {
        w_space = 0.0f;
    }
    
    // Material weight - use the interpolated max (already phase-adjusted)
    // This naturally emphasizes material more as the game progresses
    float w_mat = W_MAT_MAX;
    
    // Fragmentation
    float frag_score = std::max(0.0f, std::min(1.0f, vor.frag_score));
    float frag_term = -W_FRAG * openness * frag_score;
    
    // Frontier distance
    float max_contested_dist = (float)vor.max_contested_dist;
    float frontier_dist_term = -FRONTIER_COEFF * (1.0f - frag_score) * max_contested_dist;
    
    // Removed CLOSEST_EGG_COEFF endgame logic - not needed with high search depth
    
    float space_term = w_space * space_score;
    float mat_term = w_mat * mat_diff;
    
    float total_eval = space_term + mat_term + frag_term + frontier_dist_term;
    
    // Debug logging disabled - too verbose
    // Uncomment below and set to very high number (10000+) if needed for debugging
    /*
    static int eval_counter = 0;
    if (++eval_counter % 50000 == 0) {
        std::cerr << "EVAL_COMPONENTS my_eggs:" << my_eggs 
                  << " opp_eggs:" << opp_eggs
                  << " TOTAL_EVAL:" << total_eval << std::endl;
    }
    */
    
    return total_eval;
}

float Evaluator::get_trap_weight(const GameState& state, const VoronoiInfo& vor) {
    // Use turns_left_player instead of buggy MAX_TURNS - turn_count
    int moves_left = state.turns_left_player;
    int total_moves = MAX_TURNS;
    float phase_mat = std::max(0.0f, std::min(1.0f, (float)moves_left / total_moves));
    
    float W_SPACE_MIN = 5.0f;
    float W_SPACE_MAX = 25.0f;
    float W_MAT_MIN = 5.0f;
    float W_MAT_MAX = 25.0f;
    
    float max_contested = 8.0f;
    float openness = 0.0f;
    if (max_contested > 0) {
        openness = std::max(0.0f, std::min(1.0f, (float)vor.contested / max_contested));
    }
    
    float w_space = W_SPACE_MIN + phase_mat * (W_SPACE_MAX - W_SPACE_MIN);
    
    if (openness == 0.0f) {
        w_space = 0.0f;
    }
    
    float w_mat = W_MAT_MIN + (1.0f - phase_mat) * (W_MAT_MAX - W_MAT_MIN);
    
    return 4.0f * (w_space + w_mat);
}

float Evaluator::get_turd_weight(const GameState& state) {
    // Use turns_left_player instead of buggy MAX_TURNS - turn_count
    int moves_left = state.turns_left_player;
    int total_moves = MAX_TURNS;
    float phase_mat = std::max(0.0f, std::min(1.0f, (float)moves_left / total_moves));
    return 2.0f * phase_mat;
}

std::vector<Move> Evaluator::move_order(const GameState& state,
                                        const std::vector<Move>& moves,
                                        const VoronoiInfo& vor) {
    if (moves.empty()) return moves;
    
    std::vector<Move> filtered = moves;
    
    int total_contested = vor.contested;
    
    // Filter: if no contested squares, drop TURD moves
    if (total_contested == 0) {
        filtered.erase(
            std::remove_if(filtered.begin(), filtered.end(),
                [](const Move& m) { return m.move_type == TURD; }),
            filtered.end()
        );
    }
    
    // Filter: if any EGG move exists, drop PLAIN moves
    bool has_egg = std::any_of(filtered.begin(), filtered.end(),
        [](const Move& m) { return m.move_type == EGG; });
    if (has_egg) {
        filtered.erase(
            std::remove_if(filtered.begin(), filtered.end(),
                [](const Move& m) { return m.move_type == PLAIN; }),
            filtered.end()
        );
    }
    
    if (filtered.empty()) {
        filtered = moves;
    }
    
    // Score moves
    std::vector<std::pair<float, Move>> scored;
    
    float BASE_EGG = 3.0f;
    float BASE_PLAIN = 2.0f;
    float BASE_TURD = 1.0f;
    float DIR_WEIGHT = 0.5f;
    float TURD_CENTER_BONUS = 2.0f;
    float TURD_FRONTIER_BONUS = 3.0f;
    
    // Directional contested counts
    std::vector<int> contested_by_dir(4, 0);
    contested_by_dir[UP] = vor.contested_up;
    contested_by_dir[DOWN] = vor.contested_down;
    contested_by_dir[LEFT] = vor.contested_left;
    contested_by_dir[RIGHT] = vor.contested_right;
    
    // Context flags
    int min_frontier_dist = (vor.min_contested_dist >= 0) ? vor.min_contested_dist : 999;
    bool near_frontier = (min_frontier_dist >= 0 && min_frontier_dist <= 2);
    
    Position center(MAP_SIZE / 2, MAP_SIZE / 2);
    Position my_pos = state.chicken_player_pos;
    int my_center_dist = BitboardOps::manhattan(my_pos, center);
    
    for (const Move& mv : filtered) {
        float score = 0.0f;
        
        // Base score by move type
        if (mv.move_type == EGG) {
            score = BASE_EGG;
        } else if (mv.move_type == PLAIN) {
            score = BASE_PLAIN;
        } else {
            score = BASE_TURD;
        }
        
        // Direction bonus
        score += DIR_WEIGHT * contested_by_dir[mv.dir];
        
        // TURD buffs
        if (mv.move_type == TURD) {
            if (my_center_dist <= 2) {
                score += TURD_CENTER_BONUS;
            }
            if (near_frontier) {
                score += TURD_FRONTIER_BONUS;
            }
        }
        
        scored.push_back({score, mv});
    }
    
    // Sort by score (descending)
    std::sort(scored.begin(), scored.end(),
        [](const std::pair<float, Move>& a, const std::pair<float, Move>& b) {
            return a.first > b.first;
        });
    
    std::vector<Move> result;
    for (const auto& p : scored) {
        result.push_back(p.second);
    }
    
    return result;
}

