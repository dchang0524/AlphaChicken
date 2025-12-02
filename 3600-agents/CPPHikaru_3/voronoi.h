#ifndef VORONOI_H
#define VORONOI_H

#include "bitboard.h"
#include "bfs.h"

struct VoronoiInfo {
    int my_owned;
    int opp_owned;
    int contested;
    int max_contested_dist;
    int min_contested_dist;
    int min_egg_dist;
    int my_voronoi;
    int opp_voronoi;
    int vor_score;
    int contested_up;
    int contested_right;
    int contested_down;
    int contested_left;
    float frag_score;
};

class VoronoiAnalyzer {
public:
    static VoronoiInfo analyze(const GameState& state, Bitboard known_traps);
};

#endif // VORONOI_H

