#ifndef BFS_H
#define BFS_H

#include "bitboard.h"
#include <vector>

struct BFSResult {
    std::vector<std::vector<int>> dist_me;   // Distance from player
    std::vector<std::vector<int>> dist_opp;  // Distance from enemy
    static constexpr int UNREACHABLE = -1;
    static constexpr int BLOCKED = -2;
};

class BFS {
public:
    static BFSResult bfs_distances_both(const GameState& state, Bitboard known_traps);
    
private:
    static std::vector<std::vector<int>> bfs_single(
        const GameState& state,
        Position start,
        Bitboard opp_eggs,
        Bitboard opp_turds,
        Bitboard known_traps
    );
};

#endif // BFS_H

