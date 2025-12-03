"""
CPP Hikaru_3 Agent - Python wrapper using ctypes
"""
import ctypes
import os
from pathlib import Path
from typing import List, Tuple, Callable

# Import the game module
from game import board as board_mod
from game.enums import Direction, MoveType

# Determine the shared library path - try each platform's library
_lib_paths = [
    Path(__file__).parent / "libcpphikaru3.dylib",  # macOS (try first for local development)
    Path(__file__).parent / "libcpphikaru3.so",     # Linux
    Path(__file__).parent / "cpphikaru3.dll",       # Windows
]

_lib = None
_lib_path = None
for path in _lib_paths:
    if path.exists():
        try:
            _lib = ctypes.CDLL(str(path))
            _lib_path = path
            break
        except OSError:
            # File exists but can't be loaded (wrong architecture), try next
            continue

if _lib is None:
    raise ImportError(
        f"Could not find or load cpphikaru3 shared library. "
        f"Tried: {[str(p) for p in _lib_paths]}"
    )

# Define function signatures
_lib.init_zobrist.argtypes = [ctypes.c_int, ctypes.c_ulonglong]
_lib.init_zobrist.restype = None

_lib.create_agent.argtypes = [ctypes.c_int]
_lib.create_agent.restype = ctypes.c_void_p

_lib.destroy_agent.argtypes = [ctypes.c_void_p]
_lib.destroy_agent.restype = None

_lib.agent_play.argtypes = [
    ctypes.c_void_p,  # handle
    ctypes.c_int, ctypes.c_int,  # chicken_player_pos
    ctypes.c_int, ctypes.c_int,  # chicken_enemy_pos
    ctypes.c_int, ctypes.c_int,  # chicken_player_spawn
    ctypes.c_int, ctypes.c_int,  # chicken_enemy_spawn
    ctypes.c_int,  # player_eggs_laid
    ctypes.c_int,  # enemy_eggs_laid
    ctypes.c_int,  # player_turds_left
    ctypes.c_int,  # enemy_turds_left
    ctypes.c_int,  # player_even_chicken
    ctypes.c_int,  # enemy_even_chicken
    ctypes.c_int,  # turn_count
    ctypes.c_int,  # turns_left_player
    ctypes.c_int,  # turns_left_enemy
    ctypes.c_int,  # is_as_turn
    ctypes.c_double,  # player_time
    ctypes.c_double,  # enemy_time
    ctypes.POINTER(ctypes.c_int),  # eggs_player
    ctypes.POINTER(ctypes.c_int),  # eggs_enemy
    ctypes.POINTER(ctypes.c_int),  # turds_player
    ctypes.POINTER(ctypes.c_int),  # turds_enemy
    ctypes.POINTER(ctypes.c_int),  # found_traps
    ctypes.POINTER(ctypes.c_int),  # sensor_data
    ctypes.POINTER(ctypes.c_int),  # out_direction
    ctypes.POINTER(ctypes.c_int),  # out_move_type
    ctypes.CFUNCTYPE(ctypes.c_double),  # time_left_func
]
_lib.agent_play.restype = ctypes.c_int

_lib.agent_reset.argtypes = [ctypes.c_void_p]
_lib.agent_reset.restype = None

# Initialize Zobrist hashing
_lib.init_zobrist(8, 1234567)

class PlayerAgent:
    def __init__(self, initial_board: board_mod.Board, time_left: Callable):
        self.handle = _lib.create_agent(initial_board.game_map.MAP_SIZE)
        if not self.handle:
            raise RuntimeError("Failed to create C++ agent")
        self.known_traps = set()
        
    def __del__(self):
        if hasattr(self, 'handle') and self.handle:
            _lib.destroy_agent(self.handle)
        
    def play(
        self,
        board: board_mod.Board,
        sensor_data: List[Tuple[bool, bool]],
        time_left: Callable,
    ):
        # Update known trapdoors
        engine_found = board.found_trapdoors
        self.known_traps |= engine_found
        
        # Get board state
        my_chicken = board.chicken_player
        opp_chicken = board.chicken_enemy
        
        my_loc = my_chicken.get_location()
        opp_loc = opp_chicken.get_location()
        my_spawn = my_chicken.get_spawn()
        opp_spawn = opp_chicken.get_spawn()
        
        # Convert sets to flat arrays (x1, y1, x2, y2, ..., -1, -1)
        def set_to_array(s):
            arr = []
            for pos in s:
                arr.append(pos[0])
                arr.append(pos[1])
            arr.append(-1)  # Terminator
            arr.append(-1)
            return (ctypes.c_int * len(arr))(*arr)
        
        eggs_player_arr = set_to_array(board.eggs_player)
        eggs_enemy_arr = set_to_array(board.eggs_enemy)
        turds_player_arr = set_to_array(board.turds_player)
        turds_enemy_arr = set_to_array(board.turds_enemy)
        found_traps_arr = set_to_array(self.known_traps)
        
        # Convert sensor_data: [(heard_even, felt_even), (heard_odd, felt_odd)]
        sensor_arr = (ctypes.c_int * 4)(
            1 if sensor_data[0][0] else 0,  # heard_even
            1 if sensor_data[0][1] else 0,  # felt_even
            1 if sensor_data[1][0] else 0,  # heard_odd
            1 if sensor_data[1][1] else 0   # felt_odd
        )
        
        # Output parameters
        out_direction = ctypes.c_int()
        out_move_type = ctypes.c_int()
        
        # Create time_left callback
        # Keep a reference to prevent garbage collection
        TimeLeftFunc = ctypes.CFUNCTYPE(ctypes.c_double)
        time_left_wrapper = lambda: float(time_left())
        time_left_cb = TimeLeftFunc(time_left_wrapper)
        
        # Call C++ function
        result = _lib.agent_play(
            self.handle,
            my_loc[0], my_loc[1],
            opp_loc[0], opp_loc[1],
            my_spawn[0], my_spawn[1],
            opp_spawn[0], opp_spawn[1],
            my_chicken.eggs_laid,
            opp_chicken.eggs_laid,
            my_chicken.turds_left,
            opp_chicken.turds_left,
            my_chicken.even_chicken,
            opp_chicken.even_chicken,
            board.turn_count,
            board.turns_left_player,
            board.turns_left_enemy,
            1 if board.is_as_turn else 0,
            board.player_time,
            board.enemy_time,
            eggs_player_arr,
            eggs_enemy_arr,
            turds_player_arr,
            turds_enemy_arr,
            found_traps_arr,
            sensor_arr,
            ctypes.byref(out_direction),
            ctypes.byref(out_move_type),
            time_left_cb
        )
        
        if result != 0:
            raise RuntimeError(f"C++ agent_play returned error code {result}")
        
        # Convert result back to (Direction, MoveType) tuple
        direction = Direction(out_direction.value)
        move_type = MoveType(out_move_type.value)
        
        return (direction, move_type)
