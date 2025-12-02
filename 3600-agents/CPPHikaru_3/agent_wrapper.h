#ifndef AGENT_WRAPPER_H
#define AGENT_WRAPPER_H

#include "bitboard.h"
#include "hmm.h"
#include "search.h"
#include <vector>
#include <functional>

// Wrapper class that interfaces with Python
class CPPHikaruAgent {
public:
    CPPHikaruAgent(int map_size);
    
    // Called from Python to make a move
    // Returns move as (direction, move_type) pair
    std::pair<int, int> play(
        const GameState& state,
        const std::vector<std::pair<bool, bool>>& sensor_data,
        Bitboard known_traps,
        std::function<double()> time_left
    );
    
    // Update trapdoor beliefs from engine-found trapdoors
    void update_known_traps(const std::vector<Position>& found_traps);
    
    void reset();

private:
    TrapdoorBelief trap_belief;
    SearchEngine search_engine;
    Bitboard known_traps;
    bool traps_fully_known;
};

#endif // AGENT_WRAPPER_H

