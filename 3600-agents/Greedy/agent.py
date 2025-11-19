from collections import deque
from typing import Callable, Tuple, List
from game.enums import Direction, MoveType, loc_after_direction
from game import board
import math

from .hiddenMarkov import TrapdoorBelief


class PlayerAgent:
    def __init__(self, board: board.Board, time_left: Callable):
        # Board & trapdoor details
        self.map_size = board.game_map.MAP_SIZE
        self.trap_belief = TrapdoorBelief(self.map_size)
        self.known_traps: set[Tuple[int, int]] = set()

        # Spawn + parity (simplest correct rule)
        sx, sy = board.chicken_player.get_location()
        self.spawn = (sx, sy)
        self.egg_parity = (sx + sy) % 2

        # Optional
        self.prev_pos = None
        self.visited = [[False for _ in range(self.map_size)]
                        for _ in range(self.map_size)]

    def play(self, board: board.Board, sensor_data, time_left: Callable):
        # Update visited
        x, y = board.chicken_player.get_location()
        self.visited[x][y] = True

        # Update known trapdoors
        found = board.found_trapdoors
        new = found - self.known_traps
        for pos in new:
            self.trap_belief.set_trapdoor(pos)
        self.known_traps |= new

        # Update belief (not needed for greedy except for known traps)
        self.trap_belief.update((x, y), sensor_data)

        # Get legal moves
        moves = board.get_valid_moves()
        if not moves:
            return (Direction.STAY, MoveType.MOVE)

        # Greedy BFS movement
        mv = self.greedy_move_to_nearest_egg(board, moves)
        self.prev_pos = (x, y)
        return mv

    def build_passable_mask(self, cur_board: board.Board):
        dim = cur_board.game_map.MAP_SIZE
        passable = [[True] * dim for _ in range(dim)]

        eggs_opp = cur_board.eggs_enemy
        turds_opp = cur_board.turds_enemy
        enemy_pos = cur_board.chicken_enemy.get_location()

        blocked = set(eggs_opp) | set(turds_opp) | {enemy_pos} | self.known_traps

        # Adjacent to opponent turds
        for (tx, ty) in turds_opp:
            for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
                nx, ny = tx + dx, ty + dy
                if 0 <= nx < dim and 0 <= ny < dim:
                    blocked.add((nx, ny))

        for (bx, by) in blocked:
            if 0 <= bx < dim and 0 <= by < dim:
                passable[bx][by] = False

        return passable

    def bfs_to_targets(self, passable, targets):
        from collections import deque
        dim = len(passable)
        dist = {}
        q = deque()
        for (x, y) in targets:
            if passable[x][y]:
                dist[(x, y)] = 0
                q.append((x, y))

        while q:
            x, y = q.popleft()
            d = dist[(x, y)]
            for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
                nx, ny = x + dx, y + dy
                if not (0 <= nx < dim and 0 <= ny < dim):
                    continue
                if not passable[nx][ny]:
                    continue
                if (nx, ny) in dist:
                    continue
                dist[(nx, ny)] = d + 1
                q.append((nx, ny))

        return dist

    def move_priority(self, mv):
        # egg > turd > move
        if mv[1] == MoveType.EGG: return 2
        if mv[1] == MoveType.TURD: return 1
        return 0

    def greedy_move_to_nearest_egg(self, cur_board: board.Board, moves):
        dim = cur_board.game_map.MAP_SIZE
        my_pos = cur_board.chicken_player.get_location()

        passable = self.build_passable_mask(cur_board)

        # target squares: unclaimed egg squares matching parity
        occupied = (cur_board.eggs_player | cur_board.eggs_enemy |
                    cur_board.turds_player | cur_board.turds_enemy)

        targets = []
        for x in range(dim):
            for y in range(dim):
                if (x, y) not in occupied and (x + y) % 2 == self.egg_parity:
                    if passable[x][y]:
                        targets.append((x, y))

        if not targets:
            # fallback: prefer egg placement if possible
            egg_moves = [mv for mv in moves if mv[1] == MoveType.EGG]
            return egg_moves[0] if egg_moves else moves[0]

        # BFS distances
        dist = self.bfs_to_targets(passable, targets)

        if my_pos not in dist:
            egg_moves = [mv for mv in moves if mv[1] == MoveType.EGG]
            return egg_moves[0] if egg_moves else moves[0]

        best = None
        best_key = None

        for mv in moves:
            d, t = mv
            new_pos = loc_after_direction(my_pos, d)
            nx, ny = new_pos
            if not (0 <= nx < dim and 0 <= ny < dim):
                continue
            if not passable[nx][ny]:
                continue
            dd = dist.get(new_pos, math.inf)
            key = (-dd, self.move_priority(mv))

            if best_key is None or key > best_key:
                best_key = key
                best = mv

        return best if best else moves[0]
