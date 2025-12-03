#include "evaluation.h"
#include "game_rules.h"
#include <algorithm>
#include <cmath>
#include <iostream>

constexpr float INF = 1e8f;

float Evaluator::evaluate(const GameState& state, const VoronoiInfo& vor, 
                         const TrapdoorBelief& trap_belief) {
    // Match Python heuristics.py exactly
    if (GameRules::is_game_over(state)) {
        int my_eggs = state.player_eggs_laid;
        int opp_eggs = state.enemy_eggs_laid;
        if (my_eggs > opp_eggs) return INF;
        if (my_eggs < opp_eggs) return -INF;
        return 0.0f;
    }
    
    // --- Basic features: material and space ---
    int my_eggs = state.player_eggs_laid;
    int opp_eggs = state.enemy_eggs_laid;
    int mat_diff = my_eggs - opp_eggs;
    
    // Voronoi score: my_voronoi - opp_voronoi (already POV-relative)
    float space_score = vor.vor_score;
    
    // --- Phase terms: time-based for material, structure-based for space/risk ---
    // Time-to-go: material matters more as we approach the end.
    // Match Python exactly: moves_left = MAX_TURNS - turn_count
    int moves_left = MAX_TURNS - state.turn_count;
    int total_moves = MAX_TURNS;
    float phase_mat = std::max(0.0f, std::min(1.0f, (float)moves_left / total_moves)); // 1 early, 0 late
    
    // Openness: how much frontier there is. 0 = closed, 1 = very open.
    // There can't be more than ~8 contested squares; normalize by that.
    float max_contested = 8.0f;
    float openness = 0.0f;
    if (max_contested > 0) {
        openness = std::max(0.0f, std::min(1.0f, (float)vor.contested / max_contested));
    }
    
    // --- Weights in egg units (tunable) ---
    // Space: important when open, but never completely zero.
    float W_SPACE_MIN = 5.0f;   // closed
    float W_SPACE_MAX = 30.0f;  // very open
    
    // Material: always matters, but ramps up hard toward the end.
    float W_MAT_MIN = 5.0f;   // early
    float W_MAT_MAX = 25.0f;  // late
    
    
    // Frontier distance: how bad it is if I'm far from my most distant frontier.
    float W_FRONTIER_DIST = 1.5f;
    
    // Frontier closeness bonus.
    float FRONTIER_COEFF = 0.5f;
    
    // // --- ENDGAME CASH OUT ---
    // // If < 8 turns left, change priorities entirely.
    // if (moves_left <= 8) {
    //     // A) EGGS ARE EVERYTHING (Spike value to force laying)
    //     W_MAT_MIN = 200.0f;
    //     W_MAT_MAX = 200.0f;
        
    //     // B) SPACE IS WORTHLESS (Just a tiebreaker)
    //     W_SPACE_MIN = 0.5f;
    //     W_SPACE_MAX = 0.5f;
        
    //     // C) STOP BLOCKING
    //     // Set penalties to 0 so we stop worrying about the wall/enemy
    //     W_FRONTIER_DIST = 0.0f;
    //     FRONTIER_COEFF = 0.0f;
    // }
    
    // Interpolate weights
    // Python: w_space = W_SPACE_MIN + (1 - openness) * (W_SPACE_MAX - W_SPACE_MIN)
    float w_space = W_SPACE_MIN + (phase_mat) * (W_SPACE_MAX - W_SPACE_MIN);

    // Python: w_mat = W_MAT_MIN + (1.0 - phase_mat) * (W_MAT_MAX - W_MAT_MIN)
    float w_mat = W_MAT_MIN + (1.0f - phase_mat) * (W_MAT_MAX - W_MAT_MIN);
    
    // --- Fragmentation & frontier geometry ---
    // 1) Fragmentation score: 0 (all frontier in one blob) → 1 (highly fragmented).
    // Penalize more when the board is open.
    float frag_score = vor.frag_score;
    frag_score = std::max(0.0f, std::min(1.0f, frag_score));
    
    // 2) Max contested distance from *my chicken* to any contested square.
    // Encourage being close to the whole frontier: large distance = bad.
    float max_contested_dist = (float)vor.max_contested_dist;
    float frontier_dist_term = -FRONTIER_COEFF * (phase_mat) * max_contested_dist;
    
    // CLOSEST_EGG_COEFF from Python
    float PANIC_THRESHOLD = 8.0f;
    float CLOSEST_EGG_COEFF = 0.0f;
    if (moves_left <= PANIC_THRESHOLD || openness == 0.0f) {
        CLOSEST_EGG_COEFF = (w_mat - 5.0f) * 0.1f;
    }
    float egg_dist = (vor.min_egg_dist <= 63) ? (float)vor.min_egg_dist : 0.0f;
    
    // --- Combine everything ---
    float space_term = w_space * space_score;
    float mat_term = w_mat * mat_diff;
    
    float total_eval = space_term + mat_term + frontier_dist_term - CLOSEST_EGG_COEFF * egg_dist * 0.25f;
 
    
    return total_eval;
}

float Evaluator::get_trap_weight(const GameState& state, const VoronoiInfo& vor) {
    // Match Python exactly: moves_left = MAX_TURNS - turn_count
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
    
    
    float w_mat = W_MAT_MIN + (1.0f - phase_mat) * (W_MAT_MAX - W_MAT_MIN);
    
    return 4.0f * (w_space + w_mat);
}

float Evaluator::get_turd_weight(const GameState& state) {
    // Match Python exactly: moves_left = MAX_TURNS - turn_count
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
    
    // AGGRESSION BOOST: When behind in space (being attacked), prioritize moves toward contested areas
    float effective_dir_weight = DIR_WEIGHT;
    if (vor.vor_score < 0.0f && vor.contested > 0) {
        // Behind in space and being contested - boost directional preference significantly
        effective_dir_weight = DIR_WEIGHT * 3.0f; // Triple the weight to fight back
    }
    
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
        
        // Direction bonus (boosted when being attacked)
        score += effective_dir_weight * contested_by_dir[mv.dir];
        
        // TURD buffs
        if (mv.move_type == TURD) {
            if (my_center_dist <= 2) {
                score += TURD_CENTER_BONUS;
            }
            if (near_frontier) {
                score += TURD_FRONTIER_BONUS;
            }
        }
        
        // Tie-breaking: Add tiny deterministic offset based on move direction
        // This prevents oscillation when moves have identical scores
        // Order: RIGHT > DOWN > UP > LEFT (arbitrary but consistent)
        float tie_breaker = 0.0f;
        if (mv.dir == RIGHT) tie_breaker = 0.0003f;
        else if (mv.dir == DOWN) tie_breaker = 0.0002f;
        else if (mv.dir == UP) tie_breaker = 0.0001f;
        // LEFT gets 0.0 (lowest)
        score += tie_breaker;
        
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

