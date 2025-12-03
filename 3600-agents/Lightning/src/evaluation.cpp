#include "evaluation.h"
#include "game_rules.h"
#include <algorithm>
#include <cmath>
#include <iostream>

constexpr float INF = 1e8f;

float Evaluator::evaluate(const GameState& state, const VoronoiInfo& vor, 
                         const TrapdoorBelief& trap_belief) {
    if (GameRules::is_game_over(state)) {
        int my_eggs = state.player_eggs_laid;
        int opp_eggs = state.enemy_eggs_laid;
        return (my_eggs - opp_eggs) * 2;
    }
    
    int my_eggs = state.player_eggs_laid;
    int opp_eggs = state.enemy_eggs_laid;
    int mat_diff = my_eggs - opp_eggs;
    
    float space_score = vor.vor_score;
    
    int moves_left = state.turns_left_player;
    int total_moves = MAX_TURNS;
    float phase_mat = std::max(0.0f, std::min(1.0f, (float)moves_left / total_moves));
    int my_potential = my_eggs + std::min(0.8 * vor.my_space_egg + 0.9 * vor.my_safe_egg, moves_left/2.0);
    int opp_potential = opp_eggs + std::min(0.8 * vor.opp_space_egg + 0.9 * vor.opp_safe_egg, moves_left/2.0);

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
    
    return my_potential - opp_potential;
}

float Evaluator::get_trap_weight(const GameState& state, const VoronoiInfo& vor) {
    // Use turns_left_player instead of buggy MAX_TURNS - turn_count
    int moves_left = state.turns_left_player;
    int total_moves = MAX_TURNS;
    float phase_mat = std::max(0.0f, std::min(1.0f, (float)moves_left / total_moves));

    float max_contested = 8.0f;
    float openness = 0.0f;
    if (max_contested > 0) {
        openness = std::max(0.0f, std::min(1.0f, (float)vor.contested / max_contested));
    }
    
    return 4.0f * openness*8*2;
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

