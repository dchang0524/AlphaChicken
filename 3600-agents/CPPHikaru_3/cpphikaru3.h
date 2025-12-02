#ifndef CPPHIKARU3_H
#define CPPHIKARU3_H

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handle for the agent
typedef void* CPPHikaruAgentHandle;

// Initialize Zobrist hashing
void init_zobrist(int dim, unsigned long long seed);

// Create a new agent instance
CPPHikaruAgentHandle create_agent(int map_size);

// Destroy an agent instance
void destroy_agent(CPPHikaruAgentHandle handle);

// Play a move - returns direction and move_type as output parameters
// Returns 0 on success, -1 on error
int agent_play(
    CPPHikaruAgentHandle handle,
    // Board state
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
    // Pieces (arrays of (x,y) pairs, terminated with -1,-1)
    const int* eggs_player,
    const int* eggs_enemy,
    const int* turds_player,
    const int* turds_enemy,
    const int* found_traps,
    // Sensor data: [heard_even, felt_even, heard_odd, felt_odd]
    const int* sensor_data,
    // Output
    int* out_direction,
    int* out_move_type,
    // Time function (passed as callback)
    double (*time_left_func)(void)
);

// Reset agent state
void agent_reset(CPPHikaruAgentHandle handle);

#ifdef __cplusplus
}
#endif

#endif // CPPHIKARU3_H

