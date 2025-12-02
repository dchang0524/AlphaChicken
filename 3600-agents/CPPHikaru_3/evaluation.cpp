#include "evaluation.h"
#include "game_rules.h"
#include <algorithm>
#include <cmath>

constexpr float INF = 1e8f;

float Evaluator::evaluate(const GameState& state, const VoronoiInfo& vor, 
                         const TrapdoorBelief& trap_belief) {
    if (GameRules::is_game_over(state)) {
        int my_eggs = state.player_eggs_laid;
        int opp_eggs = state.enemy_eggs_laid;
        if (my_eggs > opp_eggs) return INF;
        if (my_eggs < opp_eggs) return -INF;
        return 0.0f;
    }
    
    int my_eggs = state.player_eggs_laid;
    int opp_eggs = state.enemy_eggs_laid;
    int mat_diff = my_eggs - opp_eggs;
    
    float space_score = vor.vor_score;
    
    // Phase terms
    int moves_left = MAX_TURNS - state.turn_count;
    int total_moves = MAX_TURNS;
    float phase_mat = std::max(0.0f, std::min(1.0f, (float)moves_left / total_moves));
    
    // Openness
    float max_contested = 8.0f;
    float openness = 0.0f;
    if (max_contested > 0) {
        openness = std::max(0.0f, std::min(1.0f, (float)vor.contested / max_contested));
    }
    
    // Weights
    float W_SPACE_MIN = 5.0f;
    float W_SPACE_MAX = 30.0f;
    float W_MAT_MIN = 5.0f;
    float W_MAT_MAX = 25.0f;
    float W_FRAG = 3.0f;
    float W_FRONTIER_DIST = 1.5f;
    float FRONTIER_COEFF = 0.5f;
    
    // Endgame (last 8 moves)
    if (moves_left <= 8) {
        W_MAT_MIN = 200.0f;
        W_MAT_MAX = 200.0f;
        W_SPACE_MIN = 0.5f;
        W_SPACE_MAX = 0.5f;
        W_FRAG = 0.0f;
        W_FRONTIER_DIST = 0.0f;
        FRONTIER_COEFF = 0.0f;
    }
    
    float w_space = W_SPACE_MIN + (1.0f - openness) * (W_SPACE_MAX - W_SPACE_MIN);
    if (openness == 0.0f) {
        w_space = 0.0f;
    }
    float w_mat = W_MAT_MIN + (1.0f - phase_mat) * (W_MAT_MAX - W_MAT_MIN);
    
    // Fragmentation
    float frag_score = std::max(0.0f, std::min(1.0f, vor.frag_score));
    float frag_term = -W_FRAG * openness * frag_score;
    
    // Frontier distance
    float max_contested_dist = (float)vor.max_contested_dist;
    float frontier_dist_term = -FRONTIER_COEFF * (1.0f - frag_score) * max_contested_dist;
    
    // Closest egg distance
    float PANIC_THRESHOLD = 8.0f;
    float CLOSEST_EGG_COEFF = 0.0f;
    if ((moves_left <= PANIC_THRESHOLD || openness == 0.0f) && moves_left > 8) {
        CLOSEST_EGG_COEFF = (w_mat - 5.0f) * 0.1f;
    }
    float egg_dist = (vor.min_egg_dist <= 63) ? (float)vor.min_egg_dist : 0.0f;
    
    float space_term = w_space * space_score;
    float mat_term = w_mat * mat_diff;
    
    return space_term + mat_term + frag_term + frontier_dist_term - CLOSEST_EGG_COEFF * egg_dist * 0.25f;
}

float Evaluator::get_trap_weight(const GameState& state, const VoronoiInfo& vor) {
    int moves_left = MAX_TURNS - state.turn_count;
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
    int moves_left = MAX_TURNS - state.turn_count;
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

