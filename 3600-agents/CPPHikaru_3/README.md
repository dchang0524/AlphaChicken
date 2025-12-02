# CPPHikaru_3

C++ optimized reimplementation of Hikaru_3 agent using bitboards and optimized game logic.

## Features

- **Bitboard representation**: Efficient 64-bit integer representation for 8x8 board
- **Optimized game rules**: Fast move generation and validation
- **Transposition tables**: Zobrist hashing for position caching
- **Voronoi analysis**: BFS-based territory evaluation
- **Hidden Markov Model**: Probabilistic trapdoor tracking
- **Negamax search**: Minimax with alpha-beta pruning and iterative deepening

## Building

### Prerequisites

- C++17 compiler (g++ or clang++)
- CMake 3.12+
- Python 3 with ctypes support (standard library)

### Build Instructions

```bash
cd CPPHikaru_3
mkdir build
cd build
cmake ..
make
```

The build will create a shared library `libcpphikaru3.so` (on Linux), `libcpphikaru3.dylib` (on macOS), or `cpphikaru3.dll` (on Windows) that can be loaded via ctypes from Python.

After building, copy or link the library to the same directory as `agent.py`:

```bash
cp build/libcpphikaru3.so .  # Linux
cp build/libcpphikaru3.dylib .  # macOS
cp build/cpphikaru3.dll .  # Windows
```

## Usage

The agent follows the same interface as other agents in the project. It will be automatically loaded by the game engine when placed in the agents directory.

The Python wrapper (`agent.py`) interfaces with the C++ implementation via ctypes, maintaining compatibility with the existing game engine while leveraging optimized C++ code for search and evaluation. The ctypes interface uses a C-compatible API exported from the C++ code.

## Implementation Details

### Bitboard Operations

- Uses `uint64_t` for 64-bit bitboards
- Efficient bit manipulation using builtin functions (`__builtin_popcountll`, `__builtin_ctzll`)
- Fast set operations (union, intersection, complement)

### Search Algorithm

- Negamax with alpha-beta pruning
- Iterative deepening with time management
- Transposition table with generation-based invalidation
- Expectimax over trapdoor scenarios

### Optimization Techniques

- Bitboard-based move generation
- Cached Voronoi analysis
- Move ordering heuristics
- Time-based depth selection

