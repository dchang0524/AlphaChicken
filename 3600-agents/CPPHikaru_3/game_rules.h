#ifndef GAME_RULES_H
#define GAME_RULES_H

#include "bitboard.h"
#include <vector>

struct Move {
    Direction dir;
    MoveType move_type;
    Move() : dir(UP), move_type(PLAIN) {}
    Move(Direction d, MoveType mt) : dir(d), move_type(mt) {}
    bool operator==(const Move& other) const {
        return dir == other.dir && move_type == other.move_type;
    }
};

// Move undo data
struct UndoData {
    Position saved_loc;
    int saved_eggs_count;
    int saved_turds_left;
    int saved_turns;
    bool saved_is_as_turn;
    bool added_to_set;
    bool triggered_trap;
    MoveType move_type;
};

class GameRules {
public:
    // Check if a move is valid
    static bool is_valid_move(const GameState& state, Direction dir, MoveType move_type, Bitboard known_traps = 0);
    
    // Get all valid moves
    static std::vector<Move> get_valid_moves(const GameState& state, Bitboard known_traps = 0);
    
    // Apply move in-place (returns undo data)
    static UndoData apply_move_inplace(GameState& state, const Move& move, 
                                      Position even_trap, Position odd_trap);
    
    // Undo move using undo data
    static void undo_move_inplace(GameState& state, const Move& move, const UndoData& undo);
    
    // Check if cell is blocked for player
    static bool is_cell_blocked(const GameState& state, Position pos, Bitboard known_traps = 0);
    
    // Check if game is over
    static bool is_game_over(const GameState& state);
    
    // Check if player has moves left
    static bool has_moves_left(const GameState& state);
    
private:
    // Helper: check if position is in enemy turd zone
    static bool is_in_enemy_turd_zone(const GameState& state, Position pos);
    
    // Helper: check if can lay egg at location
    static bool can_lay_egg_at_loc(const GameState& state, Position loc);
    
    // Helper: check if can lay turd at location
    static bool can_lay_turd_at_loc(const GameState& state, Position loc);
};

#endif // GAME_RULES_H

