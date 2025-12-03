#include "voronoi.h"
#include <algorithm>
#include <cmath>

VoronoiInfo VoronoiAnalyzer::analyze(const GameState& state, Bitboard known_traps) {
    VoronoiInfo info = {};
    info.min_contested_dist = MAP_SIZE * MAP_SIZE;
    info.min_egg_dist       = MAP_SIZE * MAP_SIZE;

    BFSResult bfs = BFS::bfs_distances_both(state, known_traps);

    bool my_even  = (state.player_even_chicken == 0);
    bool opp_even = (state.enemy_even_chicken  == 0);

    Position my_pos = state.chicken_player_pos;

    for (int x = 0; x < MAP_SIZE; ++x) {
        for (int y = 0; y < MAP_SIZE; ++y) {
            Position pos(x, y);

            int d_me  = bfs.dist_me[x][y];
            int d_opp = bfs.dist_opp[x][y];

            bool r_me  = (d_me  >= 0);
            bool r_opp = (d_opp >= 0);

            bool is_even_sq = BitboardOps::is_even_square(pos);

            // -------------------------
            // Voronoi owner (side to move = "me")
            // -------------------------
            int owner = 0; // 0 = NONE, 1 = ME, 2 = OPP

            if (r_me && (!r_opp || d_me <= d_opp)) {
                owner = 1;
                info.my_owned++;
            } else if (r_opp) {
                owner = 2;
                info.opp_owned++;
            }

            // -------------------------
            // Contested frontier (your original logic)
            // -------------------------
            if (r_me && r_opp && owner == 1 && std::abs(d_me - d_opp) <= 1) {
                info.contested++;

                if (d_me > info.max_contested_dist) {
                    info.max_contested_dist = d_me;
                }
                if (d_me < info.min_contested_dist) {
                    info.min_contested_dist = d_me;
                }

                int dx = x - my_pos.x;
                int dy = y - my_pos.y;

                if (dx != 0 || dy != 0) {
                    if (dx > 0) info.contested_right++;
                    if (dx < 0) info.contested_left++;
                    if (dy > 0) info.contested_down++;
                    if (dy < 0) info.contested_up++;
                }
            }

            // -------------------------
            // Egg metrics:
            //   - my_space_egg / my_safe_egg
            //   - opp_space_egg / opp_safe_egg
            //   "unclaimed egg of color X" = correct parity, empty square.
            // -------------------------
            bool my_unclaimed_egg  = is_unclaimed_egg(state, pos, /*for_player=*/true);
            bool opp_unclaimed_egg = is_unclaimed_egg(state, pos, /*for_player=*/false);
            int weight = BitboardOps::is_corner(pos) ? 3 : 1;

            // My-color unclaimed eggs inside my Voronoi region
            if (owner == 1 && my_unclaimed_egg) {
                if (!r_opp) {
                    info.my_safe_egg += weight;
                } else {
                    info.my_space_egg += weight;
                }
                if (r_me && d_me < info.min_egg_dist) {
                    info.min_egg_dist = d_me;
                }
            }

            // Opp-color unclaimed eggs inside opp Voronoi region
            if (owner == 2 && opp_unclaimed_egg) {
                if (!r_me) {
                    info.opp_safe_egg += weight;
                } else {
                    info.opp_space_egg += weight;
                }
            }

            // -------------------------
            // Parity-based Voronoi score (corner-weighted)
            // -------------------------

            if (is_even_sq == my_even && owner == 1) {
                info.my_voronoi += weight;
            }
            if (is_even_sq == opp_even && owner == 2) {
                info.opp_voronoi += weight;
            }
        }
    }

    info.vor_score = info.my_voronoi - info.opp_voronoi;

    // Fragmentation scaffold (left as-is, since your eval uses frag_score;
    // assume VoronoiInfo ctor zero-inits frag_score).
    std::vector<int> counts = {
        info.contested_left,
        info.contested_right,
        info.contested_up,
        info.contested_down
    };
    int total = info.contested;
    (void)counts;
    (void)total;

    if (info.min_contested_dist == MAP_SIZE * MAP_SIZE) {
        info.min_contested_dist = -1;
    }
    if (info.min_egg_dist == MAP_SIZE * MAP_SIZE) {
        info.min_egg_dist = 63; // sentinel
    }

    return info;
}
