#ifndef HMM_H
#define HMM_H

#include "bitboard.h"
#include <vector>

// Sensor kernels for trapdoor detection
struct SensorKernels {
    // prob_hear: probability of hearing at (dx, dy) from trapdoor
    static float prob_hear(int dx, int dy);
    // prob_feel: probability of feeling at (dx, dy) from trapdoor
    static float prob_feel(int dx, int dy);
};

class TrapdoorBelief {
public:
    TrapdoorBelief(int map_size);
    
    // Update belief from sensor data
    // sensor_data[0] = (heard_even, felt_even)
    // sensor_data[1] = (heard_odd, felt_odd)
    void update(Position player_pos, 
                const std::vector<std::pair<bool, bool>>& sensor_data);
    
    // Mark position as safe (stepped on it, didn't die)
    void mark_safe(Position pos);
    
    // Set trapdoor at position (collapsed belief)
    void set_trapdoor(Position pos);
    
    // Get probability at position
    float prob_at(Position pos) const;
    
    // Reset to initial distribution
    void reset();
    
    // Get potential trap positions above threshold
    std::vector<Position> get_potential_even(float threshold = 0.30f) const;
    std::vector<Position> get_potential_odd(float threshold = 0.30f) const;

private:
    int map_size;
    std::vector<std::vector<float>> p_even;  // Probability for even squares
    std::vector<std::vector<float>> p_odd;   // Probability for odd squares
    
    void init_prob();
    std::vector<std::vector<float>> get_likelihood_grid(
        Position player_pos,
        const std::vector<std::vector<float>>& kernel,
        bool sensed
    ) const;
};

#endif // HMM_H

