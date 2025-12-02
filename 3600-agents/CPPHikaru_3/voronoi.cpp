#include "voronoi.h"
#include <algorithm>
#include <cmath>

VoronoiInfo VoronoiAnalyzer::analyze(const GameState& state, Bitboard known_traps) {
    VoronoiInfo info = {};
    info.min_contested_dist = MAP_SIZE * MAP_SIZE;
    info.min_egg_dist = MAP_SIZE * MAP_SIZE;
    
    BFSResult bfs = BFS::bfs_distances_both(state, known_traps);
    
    bool my_even = (state.player_even_chicken == 0);
    bool opp_even = (state.enemy_even_chicken == 0);
    
    Position my_pos = state.chicken_player_pos;
    
    for (int x = 0; x < MAP_SIZE; ++x) {
        for (int y = 0; y < MAP_SIZE; ++y) {
            int d_me = bfs.dist_me[x][y];
            int d_opp = bfs.dist_opp[x][y];
            
            bool reachable_me = (d_me >= 0);
            bool reachable_opp = (d_opp >= 0);
            
            Position pos(x, y);
            bool is_even_sq = BitboardOps::is_even_square(pos);
            
            // Determine owner
            int owner = 0; // NONE
            if (reachable_me && (!reachable_opp || d_me <= d_opp)) {
                owner = 1; // ME
                info.my_owned++;
            } else if (reachable_opp) {
                owner = 2; // OPP
                info.opp_owned++;
            }
            
            // Contested frontier
            if (reachable_me && reachable_opp && owner == 1 && abs(d_me - d_opp) <= 1) {
                info.contested++;
                
                if (d_me > info.max_contested_dist) {
                    info.max_contested_dist = d_me;
                }
                if (d_me < info.min_contested_dist) {
                    info.min_contested_dist = d_me;
                }
                
                // Directional counts
                int dx = x - my_pos.x;
                int dy = y - my_pos.y;
                
                if (dx != 0 || dy != 0) {
                    if (dx > 0) info.contested_right++;
                    if (dx < 0) info.contested_left++;
                    if (dy > 0) info.contested_down++;
                    if (dy < 0) info.contested_up++;
                }
            }
            
            // Parity-based Voronoi (corner-weighted)
            int weight = (BitboardOps::is_corner(pos)) ? 3 : 1;
            
            if (is_even_sq == my_even && owner == 1) {
                info.my_voronoi += weight;
                if (d_me < info.min_egg_dist) {
                    info.min_egg_dist = d_me;
                }
            }
            
            if (is_even_sq == opp_even && owner == 2) {
                info.opp_voronoi += weight;
            }
        }
    }
    
    info.vor_score = info.my_voronoi - info.opp_voronoi;
    
    // Fragmentation score
    std::vector<int> counts = {info.contested_left, info.contested_right,
                               info.contested_up, info.contested_down};
    int total = info.contested;
    
    if (total <= 1) {
        info.frag_score = 0.0f;
    } else {
        int dir_count = 0;
        for (int c : counts) {
            if (c > 0) dir_count++;
        }
        
        float dir_score = (dir_count - 1) / 3.0f;
        float major_fraction = *std::max_element(counts.begin(), counts.end()) / (float)total;
        float spread_score = 1.0f - major_fraction;
        
        float opp_bonus = 0.0f;
        if (info.contested_left > 0 && info.contested_right > 0) opp_bonus += 0.5f;
        if (info.contested_up > 0 && info.contested_down > 0) opp_bonus += 0.5f;
        float opp_score = std::min(1.0f, opp_bonus);
        
        float cardinal_frag = 0.4f * spread_score + 0.2f * dir_score + 0.4f * opp_score;
        
        // Quadrant fragmentation
        std::vector<int> quad_counts(4, 0);
        for (int x = 0; x < MAP_SIZE; ++x) {
            for (int y = 0; y < MAP_SIZE; ++y) {
                int d_me = bfs.dist_me[x][y];
                int d_opp = bfs.dist_opp[x][y];
                if (d_me >= 0 && d_opp >= 0 && abs(d_me - d_opp) <= 1) {
                    int dx = x - my_pos.x;
                    int dy = y - my_pos.y;
                    if (dx > 0 && dy < 0) quad_counts[0]++;
                    else if (dx < 0 && dy < 0) quad_counts[1]++;
                    else if (dx < 0 && dy > 0) quad_counts[2]++;
                    else if (dx > 0 && dy > 0) quad_counts[3]++;
                }
            }
        }
        
        int quad_dirs = 0;
        for (int c : quad_counts) {
            if (c > 0) quad_dirs++;
        }
        float quad_spread = 1.0f - (*std::max_element(quad_counts.begin(), quad_counts.end()) / (float)total);
        float quad_score = 0.5f * quad_spread + 0.5f * ((quad_dirs - 1) / 3.0f);
        
        info.frag_score = 0.5f * cardinal_frag + 0.5f * quad_score;
        info.frag_score = std::max(0.0f, std::min(1.0f, info.frag_score));
    }
    
    if (info.min_contested_dist == MAP_SIZE * MAP_SIZE) {
        info.min_contested_dist = -1;
    }
    if (info.min_egg_dist == MAP_SIZE * MAP_SIZE) {
        info.min_egg_dist = 63;
    }
    
    return info;
}

