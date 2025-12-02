#ifndef SEARCH_H
#define SEARCH_H

#include "bitboard.h"
#include "game_rules.h"
#include "zobrist.h"
#include "voronoi.h"
#include "hmm.h"
#include "evaluation.h"
#include <unordered_map>
#include <functional>

enum TTFlag {
    TT_EXACT,
    TT_LOWER,
    TT_UPPER
};

struct TTEntry {
    float value;
    int depth;
    TTFlag flag;
    Move best_move;
    int gen;
};

struct TrapScenario {
    Position even_trap;  // -1, -1 means None
    Position odd_trap;   // -1, -1 means None
    float weight;
    bool has_even;
    bool has_odd;
};

class SearchEngine {
public:
    SearchEngine();
    
    Move search_root(const GameState& state, 
                    const TrapdoorBelief& trap_belief,
                    Bitboard known_traps,
                    int max_depth,
                    std::function<double()> time_left);
    
    void set_traps_fully_known(bool value) { traps_fully_known = value; }
    
private:
    std::unordered_map<uint64_t, TTEntry> transposition_table;
    std::unordered_map<uint64_t, VoronoiInfo> voronoi_cache;
    
    int search_gen;
    bool traps_fully_known;
    Move last_root_best;
    
    float trap_weight;
    float turd_weight;
    
    float negamax(GameState& state,
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
                  std::function<double()> time_left,
                  int root_moves_left);
    
    VoronoiInfo get_voronoi(const GameState& state, Bitboard known_traps);
    
    std::vector<TrapScenario> build_trap_scenarios(
        const TrapdoorBelief& trap_belief,
        const std::vector<Position>& potential_even,
        const std::vector<Position>& potential_odd);
    
    int choose_max_depth(const GameState& state);
};

#endif // SEARCH_H

