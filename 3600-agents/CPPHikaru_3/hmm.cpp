#include "hmm.h"
#include <cmath>
#include <algorithm>

float SensorKernels::prob_hear(int dx, int dy) {
    dx = abs(dx);
    dy = abs(dy);
    
    if (dx > 2 || dy > 2) return 0.0f;
    if (dx == 2 && dy == 2) return 0.0f;
    if (dx == 2 || dy == 2) return 0.1f;
    if (dx == 1 && dy == 1) return 0.25f;
    if (dx == 1 || dy == 1) return 0.5f;
    return 0.0f;
}

float SensorKernels::prob_feel(int dx, int dy) {
    dx = abs(dx);
    dy = abs(dy);
    
    if (dx > 1 || dy > 1) return 0.0f;
    if (dx == 1 && dy == 1) return 0.15f;
    if (dx == 1 || dy == 1) return 0.3f;
    return 0.0f;
}

TrapdoorBelief::TrapdoorBelief(int map_size) : map_size(map_size) {
    if (map_size <= 0 || map_size > 64) {
        this->map_size = 8; // Default to 8x8
    }
    p_even.assign(this->map_size, std::vector<float>(this->map_size, 0.0f));
    p_odd.assign(this->map_size, std::vector<float>(this->map_size, 0.0f));
    init_prob();
}

void TrapdoorBelief::init_prob() {
    // Initialize weights based on distance from edge
    // Edge (0) -> w=0
    // Inside edge (1) -> w=0
    // Ring 2 (2) -> w=1
    // Inner core (3+) -> w=2
    std::vector<std::vector<float>> weights(map_size, std::vector<float>(map_size, 0.0f));
    
    // Ring 2 (indices 2 to dim-3)
    if (map_size >= 6) {
        for (int x = 2; x < map_size - 2; ++x) {
            for (int y = 2; y < map_size - 2; ++y) {
                weights[x][y] = 1.0f;
            }
        }
    }
    
    // Inner core (indices 3 to dim-4)
    if (map_size >= 8) {
        for (int x = 3; x < map_size - 3; ++x) {
            for (int y = 3; y < map_size - 3; ++y) {
                weights[x][y] = 2.0f;
            }
        }
    }
    
    // Apply to even squares
    float sum_even = 0.0f;
    for (int x = 0; x < map_size; ++x) {
        for (int y = 0; y < map_size; ++y) {
            if ((x + y) % 2 == 0) {
                p_even[x][y] = weights[x][y];
                sum_even += weights[x][y];
            }
        }
    }
    
    if (sum_even > 0.0f) {
        for (int x = 0; x < map_size; ++x) {
            for (int y = 0; y < map_size; ++y) {
                p_even[x][y] /= sum_even;
            }
        }
    }
    
    // Apply to odd squares
    float sum_odd = 0.0f;
    for (int x = 0; x < map_size; ++x) {
        for (int y = 0; y < map_size; ++y) {
            if ((x + y) % 2 == 1) {
                p_odd[x][y] = weights[x][y];
                sum_odd += weights[x][y];
            }
        }
    }
    
    if (sum_odd > 0.0f) {
        for (int x = 0; x < map_size; ++x) {
            for (int y = 0; y < map_size; ++y) {
                p_odd[x][y] /= sum_odd;
            }
        }
    }
}

std::vector<std::vector<float>> TrapdoorBelief::get_likelihood_grid(
    Position player_pos,
    const std::vector<std::vector<float>>& kernel,
    bool sensed) const {
    
    // Validate inputs
    if (kernel.empty()) {
        std::vector<std::vector<float>> L(map_size, std::vector<float>(map_size, sensed ? 0.0f : 1.0f));
        return L;
    }
    
    int kernel_size = static_cast<int>(kernel.size());
    std::vector<std::vector<float>> L(map_size, std::vector<float>(map_size, sensed ? 0.0f : 1.0f));
    
    // Validate player position
    if (player_pos.x < 0 || player_pos.x >= map_size || 
        player_pos.y < 0 || player_pos.y >= map_size) {
        return L;
    }
    
    // Apply kernel
    for (int dx = -2; dx <= 2; ++dx) {
        for (int dy = -2; dy <= 2; ++dy) {
            int tx = player_pos.x - dx;
            int ty = player_pos.y - dy;
            
            if (tx >= 0 && tx < map_size && ty >= 0 && ty < map_size) {
                float kernel_val = 0.0f;
                int kx = dx + 2;
                int ky = dy + 2;
                if (kx >= 0 && kx < kernel_size && 
                    ky >= 0 && ky < static_cast<int>(kernel[kx].size())) {
                    kernel_val = kernel[kx][ky];
                }
                
                if (sensed) {
                    L[tx][ty] = kernel_val;
                } else {
                    L[tx][ty] = 1.0f - kernel_val;
                }
            }
        }
    }
    
    return L;
}

void TrapdoorBelief::update(Position player_pos,
                            const std::vector<std::pair<bool, bool>>& sensor_data) {
    if (sensor_data.size() < 2) return;
    
    // Build hear and feel kernels (5x5 for hear, 3x3 for feel)
    std::vector<std::vector<float>> hear_kernel(5, std::vector<float>(5, 0.0f));
    std::vector<std::vector<float>> feel_kernel(3, std::vector<float>(3, 0.0f));
    
    for (int dx = -2; dx <= 2; ++dx) {
        for (int dy = -2; dy <= 2; ++dy) {
            hear_kernel[dx + 2][dy + 2] = SensorKernels::prob_hear(dx, dy);
        }
    }
    
    for (int dx = -1; dx <= 1; ++dx) {
        for (int dy = -1; dy <= 1; ++dy) {
            feel_kernel[dx + 1][dy + 1] = SensorKernels::prob_feel(dx, dy);
        }
    }
    
    // Update even
    bool heard_e = sensor_data[0].first;
    bool felt_e = sensor_data[0].second;
    
    auto L_hear_e = get_likelihood_grid(player_pos, hear_kernel, heard_e);
    auto L_feel_e = get_likelihood_grid(player_pos, feel_kernel, felt_e);
    
    float total_even = 0.0f;
    for (int x = 0; x < map_size; ++x) {
        for (int y = 0; y < map_size; ++y) {
            if ((x + y) % 2 == 0) {
                p_even[x][y] *= L_hear_e[x][y] * L_feel_e[x][y];
                total_even += p_even[x][y];
            }
        }
    }
    
    if (total_even > 0.0f) {
        for (int x = 0; x < map_size; ++x) {
            for (int y = 0; y < map_size; ++y) {
                p_even[x][y] /= total_even;
            }
        }
    } else {
        // Reset to uniform
        init_prob();
    }
    
    // Update odd
    bool heard_o = sensor_data[1].first;
    bool felt_o = sensor_data[1].second;
    
    auto L_hear_o = get_likelihood_grid(player_pos, hear_kernel, heard_o);
    auto L_feel_o = get_likelihood_grid(player_pos, feel_kernel, felt_o);
    
    float total_odd = 0.0f;
    for (int x = 0; x < map_size; ++x) {
        for (int y = 0; y < map_size; ++y) {
            if ((x + y) % 2 == 1) {
                p_odd[x][y] *= L_hear_o[x][y] * L_feel_o[x][y];
                total_odd += p_odd[x][y];
            }
        }
    }
    
    if (total_odd > 0.0f) {
        for (int x = 0; x < map_size; ++x) {
            for (int y = 0; y < map_size; ++y) {
                p_odd[x][y] /= total_odd;
            }
        }
    } else {
        // Reset to uniform
        init_prob();
    }
}

void TrapdoorBelief::mark_safe(Position pos) {
    if (pos.x < 0 || pos.x >= map_size || pos.y < 0 || pos.y >= map_size) {
        return;
    }
    
    if ((pos.x + pos.y) % 2 == 0) {
        p_even[pos.x][pos.y] = 0.0f;
        float sum = 0.0f;
        for (int x = 0; x < map_size; ++x) {
            for (int y = 0; y < map_size; ++y) {
                sum += p_even[x][y];
            }
        }
        if (sum > 0.0f) {
            for (int x = 0; x < map_size; ++x) {
                for (int y = 0; y < map_size; ++y) {
                    p_even[x][y] /= sum;
                }
            }
        }
    } else {
        p_odd[pos.x][pos.y] = 0.0f;
        float sum = 0.0f;
        for (int x = 0; x < map_size; ++x) {
            for (int y = 0; y < map_size; ++y) {
                sum += p_odd[x][y];
            }
        }
        if (sum > 0.0f) {
            for (int x = 0; x < map_size; ++x) {
                for (int y = 0; y < map_size; ++y) {
                    p_odd[x][y] /= sum;
                }
            }
        }
    }
}

void TrapdoorBelief::set_trapdoor(Position pos) {
    if ((pos.x + pos.y) % 2 == 0) {
        for (int x = 0; x < map_size; ++x) {
            for (int y = 0; y < map_size; ++y) {
                p_even[x][y] = 0.0f;
            }
        }
        p_even[pos.x][pos.y] = 1.0f;
    } else {
        for (int x = 0; x < map_size; ++x) {
            for (int y = 0; y < map_size; ++y) {
                p_odd[x][y] = 0.0f;
            }
        }
        p_odd[pos.x][pos.y] = 1.0f;
    }
}

float TrapdoorBelief::prob_at(Position pos) const {
    if (pos.x < 0 || pos.x >= map_size || pos.y < 0 || pos.y >= map_size) {
        return 0.0f;
    }
    
    if ((pos.x + pos.y) % 2 == 0) {
        return p_even[pos.x][pos.y];
    } else {
        return p_odd[pos.x][pos.y];
    }
}

void TrapdoorBelief::reset() {
    init_prob();
}

std::vector<Position> TrapdoorBelief::get_potential_even(float threshold) const {
    std::vector<Position> result;
    for (int x = 0; x < map_size; ++x) {
        for (int y = 0; y < map_size; ++y) {
            if ((x + y) % 2 == 0 && p_even[x][y] >= threshold) {
                result.push_back(Position(x, y));
            }
        }
    }
    return result;
}

std::vector<Position> TrapdoorBelief::get_potential_odd(float threshold) const {
    std::vector<Position> result;
    for (int x = 0; x < map_size; ++x) {
        for (int y = 0; y < map_size; ++y) {
            if ((x + y) % 2 == 1 && p_odd[x][y] >= threshold) {
                result.push_back(Position(x, y));
            }
        }
    }
    return result;
}

