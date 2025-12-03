#include "cpphikaru3.h"
#include "agent_wrapper.h"
#include "bitboard.h"
#include "zobrist.h"
#include "search.h"
#include <vector>
#include <functional>
#include <cstring>

extern "C" {

void init_zobrist(int dim, unsigned long long seed) {
    ZobristHash::init(dim, static_cast<uint64_t>(seed));
}

CPPHikaruAgentHandle create_agent(int map_size) {
    CPPHikaruAgent* agent = new CPPHikaruAgent(map_size);
    return static_cast<CPPHikaruAgentHandle>(agent);
}

void destroy_agent(CPPHikaruAgentHandle handle) {
    if (handle) {
        CPPHikaruAgent* agent = static_cast<CPPHikaruAgent*>(handle);
        delete agent;
    }
}

int agent_play(
    CPPHikaruAgentHandle handle,
    int chicken_player_x, int chicken_player_y,
    int chicken_enemy_x, int chicken_enemy_y,
    int chicken_player_spawn_x, int chicken_player_spawn_y,
    int chicken_enemy_spawn_x, int chicken_enemy_spawn_y,
    int player_eggs_laid,
    int enemy_eggs_laid,
    int player_turds_left,
    int enemy_turds_left,
    int player_even_chicken,
    int enemy_even_chicken,
    int turn_count,
    int turns_left_player,
    int turns_left_enemy,
    int is_as_turn,
    double player_time,
    double enemy_time,
    const int* eggs_player,
    const int* eggs_enemy,
    const int* turds_player,
    const int* turds_enemy,
    const int* found_traps,
    const int* sensor_data,
    int* out_direction,
    int* out_move_type,
    double (*time_left_func)(void)) {
    
    if (!handle || !out_direction || !out_move_type) {
        return -1;
    }
    
    CPPHikaruAgent* agent = static_cast<CPPHikaruAgent*>(handle);
    
    // Build GameState
    GameState state;
    state.chicken_player_pos = Position(chicken_player_x, chicken_player_y);
    state.chicken_enemy_pos = Position(chicken_enemy_x, chicken_enemy_y);
    state.chicken_player_spawn = Position(chicken_player_spawn_x, chicken_player_spawn_y);
    state.chicken_enemy_spawn = Position(chicken_enemy_spawn_x, chicken_enemy_spawn_y);
    state.player_eggs_laid = player_eggs_laid;
    state.enemy_eggs_laid = enemy_eggs_laid;
    state.player_turds_left = player_turds_left;
    state.enemy_turds_left = enemy_turds_left;
    state.player_even_chicken = player_even_chicken;
    state.enemy_even_chicken = enemy_even_chicken;
    state.turn_count = turn_count;
    state.turns_left_player = turns_left_player;
    state.turns_left_enemy = turns_left_enemy;
    state.is_as_turn = (is_as_turn != 0);
    state.player_time = player_time;
    state.enemy_time = enemy_time;
    
    // Convert piece arrays to bitboards (arrays are terminated with -1,-1)
    state.eggs_player = 0;
    if (eggs_player) {
        for (int i = 0; eggs_player[i*2] != -1 && eggs_player[i*2+1] != -1; i++) {
            int x = eggs_player[i*2];
            int y = eggs_player[i*2 + 1];
            if (BitboardOps::is_valid(x, y)) {
                state.eggs_player = BitboardOps::set_bit(state.eggs_player, x, y);
            }
        }
    }
    
    state.eggs_enemy = 0;
    if (eggs_enemy) {
        for (int i = 0; eggs_enemy[i*2] != -1 && eggs_enemy[i*2+1] != -1; i++) {
            int x = eggs_enemy[i*2];
            int y = eggs_enemy[i*2 + 1];
            if (BitboardOps::is_valid(x, y)) {
                state.eggs_enemy = BitboardOps::set_bit(state.eggs_enemy, x, y);
            }
        }
    }
    
    state.turds_player = 0;
    if (turds_player) {
        for (int i = 0; turds_player[i*2] != -1 && turds_player[i*2+1] != -1; i++) {
            int x = turds_player[i*2];
            int y = turds_player[i*2 + 1];
            if (BitboardOps::is_valid(x, y)) {
                state.turds_player = BitboardOps::set_bit(state.turds_player, x, y);
            }
        }
    }
    
    state.turds_enemy = 0;
    if (turds_enemy) {
        for (int i = 0; turds_enemy[i*2] != -1 && turds_enemy[i*2+1] != -1; i++) {
            int x = turds_enemy[i*2];
            int y = turds_enemy[i*2 + 1];
            if (BitboardOps::is_valid(x, y)) {
                state.turds_enemy = BitboardOps::set_bit(state.turds_enemy, x, y);
            }
        }
    }
    
    // Convert found traps
    std::vector<Position> found_traps_vec;
    Bitboard known_traps = 0;
    if (found_traps) {
        for (int i = 0; found_traps[i*2] != -1 && found_traps[i*2+1] != -1; i++) {
            int x = found_traps[i*2];
            int y = found_traps[i*2 + 1];
            if (BitboardOps::is_valid(x, y)) {
                Position pos(x, y);
                found_traps_vec.push_back(pos);
                known_traps = BitboardOps::set_bit(known_traps, x, y);
            }
        }
    }
    
    // Convert sensor data: [heard_even, felt_even, heard_odd, felt_odd]
    std::vector<std::pair<bool, bool>> sensor_list;
    if (sensor_data) {
        sensor_list.push_back({sensor_data[0] != 0, sensor_data[1] != 0});
        sensor_list.push_back({sensor_data[2] != 0, sensor_data[3] != 0});
    }
    
    // Update known traps
    agent->update_known_traps(found_traps_vec);
    
    // Create time_left function wrapper
    std::function<double()> time_left_wrapper = [time_left_func]() -> double {
        return time_left_func ? time_left_func() : 360.0;
    };
    
    // Call play
    auto result = agent->play(state, sensor_list, known_traps, time_left_wrapper);
    
    *out_direction = result.first;
    *out_move_type = result.second;
    
    return 0;
}

void agent_reset(CPPHikaruAgentHandle handle) {
    if (handle) {
        CPPHikaruAgent* agent = static_cast<CPPHikaruAgent*>(handle);
        agent->reset();
    }
}

} // extern "C"

