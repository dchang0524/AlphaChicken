from typing import List, Tuple

import numpy as np

from game.chicken import Chicken
from game.enums import Cell, Direction, MoveType, Result, WinReason
from game.game_map import GameMap, prob_feel, prob_hear
from game.history import History


def delta_locs(loc_a: Tuple[int, int], loc_b: Tuple[int, int]) -> Tuple[int, int]:
    ax, ay = loc_a
    bx, by = loc_b
    return (abs(ax - bx), abs(ay - by))


def choose_trapdoor(weights, parity) -> Tuple[int, int]:
    dim = weights.shape[0]
    flattened = weights.flatten()
    i = 1
    j = parity
    while (i + j) % 2 != parity:
        indx = np.random.choice(len(flattened), p=flattened)
        i = indx // dim
        j = indx % dim
    return (i, j)


class TrapdoorManager:
    def __init__(self, game_map: GameMap):
        self.game_map = game_map
        self.spawns = []
        self.trapdoors = []

    # choose_spawns should be called before choose trapdoor_weights
    def choose_spawns(self):
        if len(self.spawns) > 0:
            print("ERROR: choose_spawns called twice")
        dim = self.game_map.MAP_SIZE
        edge = np.random.randint(0, 4)
        if edge == 0:
            i = np.random.randint(1, dim - 1)
            j = 0
        elif edge == 1:
            i = np.random.randint(1, dim - 1)
            j = dim - 1
        elif edge == 2:
            i = 0
            j = np.random.randint(1, dim - 1)
        else:
            i = dim - 1
            j = np.random.randint(1, dim - 1)

        # First is random even, second is antipodal
        if (i + j) % 2 == 0:
            even_spawn = (i, j)

            if(edge <2):
                odd_spawn = (i, dim - 1 - j)
            else:
                odd_spawn = (dim - 1 - i, j)
        else:
            odd_spawn = (i, j)
            if(edge <2):
                even_spawn = (i, dim - 1 - j)
            else:
                even_spawn = (dim - 1 - i, j)
        self.spawns = [even_spawn, odd_spawn]

        return self.spawns.copy()

    def choose_trapdoors(self):
        if len(self.trapdoors) > 0:
            print("ERROR: choose_trapdoors called twice")

        dim = self.game_map.MAP_SIZE
        unnormalized = np.zeros((dim, dim))
        unnormalized[2 : dim - 2, 2 : dim - 2] = 1.0
        unnormalized[3 : dim - 3, 3 : dim - 3] = 2.0
        normalized = unnormalized / np.sum(unnormalized)
        even_trapdoor = choose_trapdoor(normalized, 0)
        odd_trapdoor = choose_trapdoor(normalized, 1)
        self.trapdoors = [even_trapdoor, odd_trapdoor]
        return self.trapdoors.copy()

    def sample_trapdoors(self, loc: Tuple[int, int]) -> List[Tuple[bool, bool]]:
        result = []
        for door_indx in range(2):
            door = self.trapdoors[door_indx]
            delta = delta_locs(loc, door)
            hear_p = prob_hear(delta[0], delta[1])
            did_hear = np.random.rand() < hear_p
            feel_p = prob_feel(delta[0], delta[1])
            did_feel = np.random.rand() < feel_p
            result.append((did_hear, did_feel))
        return result

    def is_trapdoor(self, loc: Tuple[int, int]) -> bool:
        return loc in self.trapdoors

    def get_trapdoors(self):
        return self.trapdoors.copy()

    def get_spawns(self):
        return self.spawns.copy()
