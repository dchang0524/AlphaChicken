#ifndef EVALUATION_H
#define EVALUATION_H

#include "bitboard.h"
#include "voronoi.h"
#include "hmm.h"
#include "game_rules.h"

class Evaluator {
public:
    static float evaluate(const GameState& state, const VoronoiInfo& vor, 
                         const TrapdoorBelief& trap_belief,
                         int root_moves_left,
                         Bitboard known_traps);
    
    static std::vector<Move> move_order(const GameState& state, 
                                       const std::vector<Move>& moves,
                                       const VoronoiInfo& vor);
    
    static float get_trap_weight(const GameState& state, const VoronoiInfo& vor);
    static float get_turd_weight(const GameState& state);
};

#endif // EVALUATION_H

