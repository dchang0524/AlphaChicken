#include "game_rules.h"
#include <algorithm>

bool GameRules::is_in_enemy_turd_zone(const GameState& state, Position pos) {
    Bitboard pos_bb = state.pos_to_bitboard(pos);
    Bitboard turd_zone = BitboardOps::get_turd_zone(state.turds_enemy);
    return (pos_bb & turd_zone) != 0;
}

bool GameRules::is_cell_blocked(const GameState& state, Position pos) {
    if (!BitboardOps::is_valid(pos.x, pos.y)) {
        return true;
    }
    
    Bitboard pos_bb = state.pos_to_bitboard(pos);
    
    
    // Blocked by enemy chicken
    if (pos == state.chicken_enemy_pos) {
        return true;
    }
    
    // Blocked by enemy eggs
    if ((pos_bb & state.eggs_enemy) != 0) {
        return true;
    }
    
    // Blocked by enemy turd zone
    if (is_in_enemy_turd_zone(state, pos)) {
        return true;
    }
    
    return false;
}

bool GameRules::can_lay_egg_at_loc(const GameState& state, Position loc) {
    // Check parity
    bool is_even = BitboardOps::is_even_square(loc);
    if (is_even != (state.player_even_chicken == 0)) {
        return false;
    }
    
    Bitboard loc_bb = state.pos_to_bitboard(loc);
    
    // Can't lay on existing pieces
    if ((loc_bb & state.eggs_player) != 0) return false;
    if ((loc_bb & state.turds_player) != 0) return false;
    if ((loc_bb & state.turds_enemy) != 0) return false;
    
    return true;
}

bool GameRules::can_lay_turd_at_loc(const GameState& state, Position loc) {
    // Must have turds left
    if (state.player_turds_left <= 0) {
        return false;
    }
    
    // Must be at least 2 Manhattan distance from enemy
    if (BitboardOps::manhattan(loc, state.chicken_enemy_pos) < 2) {
        return false;
    }
    
    Bitboard loc_bb = state.pos_to_bitboard(loc);
    
    // Can't place on existing pieces
    if ((loc_bb & state.turds_player) != 0) return false;
    if ((loc_bb & state.turds_enemy) != 0) return false;
    if ((loc_bb & state.eggs_player) != 0) return false;
    if ((loc_bb & state.eggs_enemy) != 0) return false;
    
    return true;
}

bool GameRules::is_valid_move(const GameState& state, Direction dir, MoveType move_type) {
    // Check turds left
    if (move_type == TURD && state.player_turds_left <= 0) {
        return false;
    }
    
    Position my_loc = state.chicken_player_pos;
    
    // Check parity for eggs
    if (move_type == EGG) {
        bool is_even = BitboardOps::is_even_square(my_loc);
        if (is_even != (state.player_even_chicken == 0)) {
            return false;
        }
    }
    
    // Check destination - check for enemy chicken, eggs, and turd zones (but NOT known traps)
    Position new_loc = BitboardOps::loc_after_direction(my_loc, dir);
    if (is_cell_blocked(state, new_loc)) {  // Pass 0 to ignore known_traps
        return false;
    }
    
    
    // For EGG/TURD moves, check current square
    if (move_type != PLAIN) {
        Bitboard my_loc_bb = state.pos_to_bitboard(my_loc);
        if ((my_loc_bb & state.eggs_player) != 0) return false;
        if ((my_loc_bb & state.turds_player) != 0) return false;
    }
    
    // For TURD, check distance
    if (move_type == TURD) {
        if (!can_lay_turd_at_loc(state, my_loc)) {
            return false;
        }
    }
    
    // For EGG, check can lay
    if (move_type == EGG) {
        if (!can_lay_egg_at_loc(state, my_loc)) {
            return false;
        }
    }
    
    return true;
}

std::vector<Move> GameRules::get_valid_moves(const GameState& state, Bitboard known_traps) {
    std::vector<Move> moves;
    
    for (int d = 0; d < 4; ++d) {
        Direction dir = static_cast<Direction>(d);
        for (int mt = 0; mt < 3; ++mt) {
            MoveType move_type = static_cast<MoveType>(mt);
            if (is_valid_move(state, dir, move_type)) {
                moves.push_back(Move(dir, move_type));
            }
        }
    }
    
    return moves;
}

UndoData GameRules::apply_move_inplace(GameState& state, const Move& move,
                                       Position even_trap, Position odd_trap) {
    UndoData undo;
    
    Direction dir = move.dir;
    MoveType move_type = move.move_type;
    
    undo.saved_loc = state.chicken_player_pos;
    undo.saved_eggs_count = state.player_eggs_laid;
    undo.saved_turds_left = state.player_turds_left;
    undo.saved_turns = state.turns_left_player;
    undo.saved_is_as_turn = state.is_as_turn;
    undo.move_type = move_type;
    undo.added_to_set = false;
    undo.triggered_trap = false;
    undo.blocked_bonus_applied = false;
    undo.saved_enemy_eggs_count = state.enemy_eggs_laid;
    
    Position my_loc = state.chicken_player_pos;
    Bitboard my_loc_bb = state.pos_to_bitboard(my_loc);
    
    // Handle EGG move
    if (move_type == EGG) {
        int reward = BitboardOps::is_corner(my_loc) ? CORNER_REWARD : 1;
        state.player_eggs_laid += reward;
        state.eggs_player |= my_loc_bb;
        undo.added_to_set = true;
    }
    // Handle TURD move
    else if (move_type == TURD) {
        state.player_turds_left--;
        state.turds_player |= my_loc_bb;
        undo.added_to_set = true;
    }
    
    // Move chicken
    Position new_loc = BitboardOps::loc_after_direction(my_loc, dir);
    state.chicken_player_pos = new_loc;
    
    // Check trapdoor
    if ((new_loc == even_trap && BitboardOps::is_even_square(new_loc)) ||
        (new_loc == odd_trap && !BitboardOps::is_even_square(new_loc))) {
        undo.triggered_trap = true;
        state.enemy_eggs_laid += TRAPDOOR_PENALTY;
        state.chicken_player_pos = state.chicken_player_spawn;
    }
    
    // Update turn
    // NOTE: turn_count is NOT incremented during search (matches Python agent)
    // It stays constant at the root value for phase calculations
    state.turns_left_player--;
    state.is_as_turn = !state.is_as_turn;
    
    // Reverse perspective (swap player/enemy)
    std::swap(state.eggs_player, state.eggs_enemy);
    std::swap(state.turds_player, state.turds_enemy);
    std::swap(state.chicken_player_pos, state.chicken_enemy_pos);
    std::swap(state.chicken_player_spawn, state.chicken_enemy_spawn);
    std::swap(state.player_eggs_laid, state.enemy_eggs_laid);
    std::swap(state.player_turds_left, state.enemy_turds_left);
    std::swap(state.player_even_chicken, state.enemy_even_chicken);
    std::swap(state.turns_left_player, state.turns_left_enemy);
    std::swap(state.player_time, state.enemy_time);
    
    // Check for blocking: if the new player (former enemy) has no moves left
    // AND they still have turns left, give +5 eggs bonus to the new enemy (former player)
    // This matches Python's end_turn() logic: if not self.has_moves_left(enemy=True)
    if (!has_moves_left(state) && state.turns_left_player > 0) {
        // Give +5 eggs to enemy (the player who just made the move, now enemy after swap)
        state.enemy_eggs_laid += 5;
        undo.blocked_bonus_applied = true;
    }
    
    return undo;
}

void GameRules::undo_move_inplace(GameState& state, const Move& move, const UndoData& undo) {
    // Undo blocking bonus FIRST (before perspective swap, while still in swapped perspective)
    // The bonus was given to enemy_eggs_laid after the perspective swap
    if (undo.blocked_bonus_applied) {
        state.enemy_eggs_laid -= 5;
    }
    
    // Reverse perspective (undo the swap)
    std::swap(state.eggs_player, state.eggs_enemy);
    std::swap(state.turds_player, state.turds_enemy);
    std::swap(state.chicken_player_pos, state.chicken_enemy_pos);
    std::swap(state.chicken_player_spawn, state.chicken_enemy_spawn);
    std::swap(state.player_eggs_laid, state.enemy_eggs_laid);
    std::swap(state.player_turds_left, state.enemy_turds_left);
    std::swap(state.player_even_chicken, state.enemy_even_chicken);
    std::swap(state.turns_left_player, state.turns_left_enemy);
    std::swap(state.player_time, state.enemy_time);
    
    // Restore state
    state.chicken_player_pos = undo.saved_loc;
    state.player_eggs_laid = undo.saved_eggs_count;
    state.player_turds_left = undo.saved_turds_left;
    state.turns_left_player = undo.saved_turns;
    state.is_as_turn = undo.saved_is_as_turn;
    
    // Remove piece if added
    if (undo.added_to_set) {
        Bitboard my_loc_bb = state.pos_to_bitboard(undo.saved_loc);
        if (undo.move_type == EGG) {
            state.eggs_player &= ~my_loc_bb;
        } else if (undo.move_type == TURD) {
            state.turds_player &= ~my_loc_bb;
        }
    }
    
    // Undo trapdoor effect
    if (undo.triggered_trap) {
        state.enemy_eggs_laid -= TRAPDOOR_PENALTY;
    }
}

bool GameRules::is_game_over(const GameState& state) {
    // Game is over when:
    // 1. Turns run out (both players exhausted their turns)
    // 2. A player is blocked (has no moves left but still has turns left)
    if (state.turns_left_player <= 0 || state.turns_left_enemy <= 0) {
        return true;
    }
    
    // Check if current player is blocked (no moves left but still has turns)
    // This matches Python's chicken_blocked logic
    if (!has_moves_left(state) && state.turns_left_player > 0) {
        return true;
    }
    
    return false;
}

bool GameRules::has_moves_left(const GameState& state) {
    return !get_valid_moves(state, 0).empty();  // Pass 0 for known_traps (not used in has_moves_left check)
}

