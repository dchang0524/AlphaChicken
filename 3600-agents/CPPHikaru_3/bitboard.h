#ifndef BITBOARD_H
#define BITBOARD_H

#include <cstdint>
#include <cstdlib>
#include <vector>

// Constants
constexpr int MAP_SIZE = 8;
constexpr int MAP_AREA = MAP_SIZE * MAP_SIZE;
constexpr int MAX_TURNS = 40;
constexpr int MAX_TURDS = 5;
constexpr int CORNER_REWARD = 3;
constexpr int TRAPDOOR_PENALTY = 4;

// Directions
enum Direction {
    UP = 0,
    RIGHT = 1,
    DOWN = 2,
    LEFT = 3
};

// Move Types
enum MoveType {
    PLAIN = 0,
    EGG = 1,
    TURD = 2
};

// Bitboard type - represents a set of squares
using Bitboard = uint64_t;

// Position representation
struct Position {
    int x, y;
    Position() : x(0), y(0) {}
    Position(int x, int y) : x(x), y(y) {}
    bool operator==(const Position& other) const {
        return x == other.x && y == other.y;
    }
    bool operator<(const Position& other) const {
        if (x != other.x) return x < other.x;
        return y < other.y;
    }
};

// Bitboard utilities
namespace BitboardOps {
    // Convert position to bitboard index (x, y) -> bit
    inline int pos_to_bit(int x, int y) {
        return y * MAP_SIZE + x;
    }
    
    // Convert bit to position
    inline Position bit_to_pos(int bit) {
        return Position(bit % MAP_SIZE, bit / MAP_SIZE);
    }
    
    // Get bit at position
    inline bool get_bit(Bitboard bb, int x, int y) {
        int bit = pos_to_bit(x, y);
        return (bb >> bit) & 1;
    }
    
    // Set bit at position
    inline Bitboard set_bit(Bitboard bb, int x, int y) {
        int bit = pos_to_bit(x, y);
        return bb | (1ULL << bit);
    }
    
    // Clear bit at position
    inline Bitboard clear_bit(Bitboard bb, int x, int y) {
        int bit = pos_to_bit(x, y);
        return bb & ~(1ULL << bit);
    }
    
    // Count bits
    inline int popcount(Bitboard bb) {
        return __builtin_popcountll(bb);
    }
    
    // Get first set bit
    inline int bsf(Bitboard bb) {
        return __builtin_ctzll(bb);
    }
    
    // Shift operations
    inline Bitboard shift_north(Bitboard bb) {
    return bb >> MAP_SIZE;
}

    inline Bitboard shift_south(Bitboard bb) {
        return bb << MAP_SIZE;
    }
    
    inline Bitboard shift_east(Bitboard bb) {
        return (bb << 1) & 0xFEFEFEFEFEFEFEFEULL; // Remove overflow, zeros out x = 0
    }
    
    inline Bitboard shift_west(Bitboard bb) {
        return (bb >> 1) & 0x7F7F7F7F7F7F7F7FULL; // Remove underflow, zeros out x = 7
    }
    
    // Get neighbors (4-directional)
    inline Bitboard get_neighbors(Bitboard bb) {
        return shift_north(bb) | shift_south(bb) | shift_east(bb) | shift_west(bb);
    }
    
    // Get squares adjacent to turds (including turd squares themselves)
    inline Bitboard get_turd_zone(Bitboard turds) {
        return turds | get_neighbors(turds);
    }
    
    // Check if position is on board
    inline bool is_valid(int x, int y) {
        return x >= 0 && x < MAP_SIZE && y >= 0 && y < MAP_SIZE;
    }
    
    // Get position after direction
    inline Position loc_after_direction(Position pos, Direction dir) {
        switch (dir) {
            case UP:    return Position(pos.x, pos.y - 1);
            case DOWN:  return Position(pos.x, pos.y + 1);
            case LEFT:  return Position(pos.x - 1, pos.y);
            case RIGHT: return Position(pos.x + 1, pos.y);
            default:    return pos;
        }
    }
    
    // Manhattan distance
    inline int manhattan(Position a, Position b) {
        return abs(a.x - b.x) + abs(a.y - b.y);
    }
    
    // Check if corner
    inline bool is_corner(Position pos) {
        return (pos.x == 0 || pos.x == MAP_SIZE - 1) &&
               (pos.y == 0 || pos.y == MAP_SIZE - 1);
    }
    
    // Check parity
    inline bool is_even_square(Position pos) {
        return (pos.x + pos.y) % 2 == 0;
    }
}

// Game state representation
struct GameState {
    // Bitboards
    Bitboard eggs_player;
    Bitboard eggs_enemy;
    Bitboard turds_player;
    Bitboard turds_enemy;
    Bitboard known_traps;
    
    // Chicken positions
    Position chicken_player_pos;
    Position chicken_enemy_pos;
    Position chicken_player_spawn;
    Position chicken_enemy_spawn;
    
    // Chicken state
    int player_eggs_laid;
    int enemy_eggs_laid;
    int player_turds_left;
    int enemy_turds_left;
    int player_even_chicken; // 0 or 1
    int enemy_even_chicken;  // 0 or 1
    
    // Game metadata
    int turn_count;
    int turns_left_player;
    int turns_left_enemy;
    bool is_as_turn; // true if it's player A's turn
    double player_time;
    double enemy_time;
    
    GameState() {
        eggs_player = 0;
        eggs_enemy = 0;
        turds_player = 0;
        turds_enemy = 0;
        known_traps = 0;
        player_eggs_laid = 0;
        enemy_eggs_laid = 0;
        player_turds_left = MAX_TURDS;
        enemy_turds_left = MAX_TURDS;
        turn_count = 0;
        turns_left_player = MAX_TURNS;
        turns_left_enemy = MAX_TURNS;
        is_as_turn = true;
        player_time = 360.0;
        enemy_time = 360.0;
    }
    
    // Helper to get bitboard from position
    inline Bitboard pos_to_bitboard(Position pos) const {
        if (!BitboardOps::is_valid(pos.x, pos.y)) return 0;
        return BitboardOps::set_bit(0, pos.x, pos.y);
    }
};

#endif // BITBOARD_H

