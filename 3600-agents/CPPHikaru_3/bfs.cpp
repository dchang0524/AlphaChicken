#include "bfs.h"
#include <queue>
#include <algorithm>

std::vector<std::vector<int>> BFS::bfs_single(
    const GameState& state,
    Position start,
    Bitboard opp_eggs,
    Bitboard opp_turds,
    Bitboard known_traps) {
    
    std::vector<std::vector<int>> dist(
        MAP_SIZE, std::vector<int>(MAP_SIZE, BFSResult::UNREACHABLE)
    );
    
    // Mark blocked cells
    Bitboard blocked = opp_eggs | opp_turds | known_traps;
    // Add turd zones (adjacent to opponent turds)
    Bitboard turd_zone = BitboardOps::get_turd_zone(opp_turds);
    blocked |= turd_zone;
    
    // Mark blocked in dist array
    for (int y = 0; y < MAP_SIZE; ++y) {
        for (int x = 0; x < MAP_SIZE; ++x) {
            if (BitboardOps::get_bit(blocked, x, y)) {
                dist[x][y] = BFSResult::BLOCKED;
            }
        }
    }
    
    // If start is blocked, return
    if (dist[start.x][start.y] == BFSResult::BLOCKED) {
        return dist;
    }
    
    // BFS
    std::queue<Position> q;
    dist[start.x][start.y] = 0;
    q.push(start);
    
    static const int dx[] = {0, 1, 0, -1};
    static const int dy[] = {-1, 0, 1, 0};
    
    while (!q.empty()) {
        Position current = q.front();
        q.pop();
        int d = dist[current.x][current.y];
        
        for (int i = 0; i < 4; ++i) {
            Position next(current.x + dx[i], current.y + dy[i]);
            
            if (!BitboardOps::is_valid(next.x, next.y)) {
                continue;
            }
            
            // Only process unreachable cells (skip blocked and already visited)
            if (dist[next.x][next.y] != BFSResult::UNREACHABLE) {
                continue;
            }
            
            dist[next.x][next.y] = d + 1;
            q.push(next);
        }
    }
    
    return dist;
}

BFSResult BFS::bfs_distances_both(const GameState& state, Bitboard known_traps) {
    BFSResult result;
    
    // Player BFS: blocked by opponent pieces
    result.dist_me = bfs_single(
        state,
        state.chicken_player_pos,
        state.eggs_enemy,
        state.turds_enemy,
        known_traps
    );
    
    // Enemy BFS: blocked by our pieces
    result.dist_opp = bfs_single(
        state,
        state.chicken_enemy_pos,
        state.eggs_player,
        state.turds_player,
        known_traps
    );
    
    return result;
}

