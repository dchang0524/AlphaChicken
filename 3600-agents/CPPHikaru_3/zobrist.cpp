#include "zobrist.h"
#include <algorithm>
#include <cstdlib>

uint64_t ZobristHash::table[MAP_SIZE][MAP_SIZE][Z_NUM_FEATURES];
uint64_t ZobristHash::side_to_move = 0;
bool ZobristHash::initialized = false;

uint64_t ZobristHash::random_uint64(uint64_t& seed) {
    seed ^= seed << 13;
    seed ^= seed >> 7;
    seed ^= seed << 17;
    return seed;
}

void ZobristHash::init(int dim, uint64_t seed) {
    if (initialized) return;
    
    uint64_t rng_state = seed;
    
    for (int x = 0; x < MAP_SIZE; ++x) {
        for (int y = 0; y < MAP_SIZE; ++y) {
            for (int f = 0; f < Z_NUM_FEATURES; ++f) {
                table[x][y][f] = random_uint64(rng_state);
            }
        }
    }
    
    side_to_move = random_uint64(rng_state);
    initialized = true;
}

uint64_t ZobristHash::hash(const GameState& state, Bitboard known_traps) {
    if (!initialized) {
        init(MAP_SIZE);
    }
    
    uint64_t h = 0;
    
    // Hash chickens
    Position my_pos = state.chicken_player_pos;
    h ^= table[my_pos.x][my_pos.y][Z_ME_CHICKEN];
    
    Position opp_pos = state.chicken_enemy_pos;
    h ^= table[opp_pos.x][opp_pos.y][Z_OPP_CHICKEN];
    
    // Hash eggs (iterate through set bits)
    Bitboard eggs = state.eggs_player;
    while (eggs) {
        int bit = BitboardOps::bsf(eggs);
        Position pos = BitboardOps::bit_to_pos(bit);
        h ^= table[pos.x][pos.y][Z_ME_EGG];
        eggs &= eggs - 1; // Clear least significant bit
    }
    
    eggs = state.eggs_enemy;
    while (eggs) {
        int bit = BitboardOps::bsf(eggs);
        Position pos = BitboardOps::bit_to_pos(bit);
        h ^= table[pos.x][pos.y][Z_OPP_EGG];
        eggs &= eggs - 1;
    }
    
    // Hash turds
    Bitboard turds = state.turds_player;
    while (turds) {
        int bit = BitboardOps::bsf(turds);
        Position pos = BitboardOps::bit_to_pos(bit);
        h ^= table[pos.x][pos.y][Z_ME_TURD];
        turds &= turds - 1;
    }
    
    turds = state.turds_enemy;
    while (turds) {
        int bit = BitboardOps::bsf(turds);
        Position pos = BitboardOps::bit_to_pos(bit);
        h ^= table[pos.x][pos.y][Z_OPP_TURD];
        turds &= turds - 1;
    }
    
    // Hash known traps
    while (known_traps) {
        int bit = BitboardOps::bsf(known_traps);
        Position pos = BitboardOps::bit_to_pos(bit);
        h ^= table[pos.x][pos.y][Z_KNOWN_TRAP];
        known_traps &= known_traps - 1;
    }
    
    // Side to move
    h ^= side_to_move;
    
    return h;
}

