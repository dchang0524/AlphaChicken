#ifndef ZOBRIST_H
#define ZOBRIST_H

#include "bitboard.h"
#include <cstdint>
#include <unordered_map>

// Zobrist feature indices
enum {
    Z_ME_EGG = 0,
    Z_OPP_EGG = 1,
    Z_ME_TURD = 2,
    Z_OPP_TURD = 3,
    Z_ME_CHICKEN = 4,
    Z_OPP_CHICKEN = 5,
    Z_KNOWN_TRAP = 6,
    Z_NUM_FEATURES = 7
};

class ZobristHash {
public:
    static void init(int dim, uint64_t seed = 1234567);
    static uint64_t hash(const GameState& state, Bitboard known_traps);
    
private:
    static uint64_t table[MAP_SIZE][MAP_SIZE][Z_NUM_FEATURES];
    static uint64_t side_to_move;
    static bool initialized;
    
    static uint64_t random_uint64(uint64_t& seed);
};

#endif // ZOBRIST_H

