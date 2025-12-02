# Building on Linux Server

Since you're on macOS and need a `.so` file for Linux, you'll need to compile on the Linux server.

## Quick Steps

### 1. Prepare the zip file (on your Mac)

```bash
cd 3600-agents/CPPHikaru_3
./build_for_linux.sh
```

This creates `CPPHikaru_3.zip` with all source files (excluding macOS binaries).

### 2. Upload to server and compile

```bash
# On Linux server
unzip CPPHikaru_3.zip
cd CPPHikaru_3
make
```

This will create `libcpphikaru3.so` that you can use.

## Alternative: Compile with g++ directly

If `make` doesn't work, you can compile manually:

```bash
g++ -std=c++17 -O3 -march=native -Wall -Wextra -fPIC -shared \
    cpphikaru3.cpp agent_wrapper.cpp search.cpp evaluation.cpp \
    voronoi.cpp bfs.cpp hmm.cpp zobrist.cpp game_rules.cpp \
    -o libcpphikaru3.so
```

## Verify the library

After building, verify it's a Linux library:

```bash
file libcpphikaru3.so
# Should show: ELF 64-bit LSB shared object, x86-64, version 1 (SYSV), dynamically linked...
```

## Requirements on Linux Server

The server needs:
- `g++` or `clang++` (C++17 support)
- Standard libraries (no special dependencies needed)

Check with:
```bash
g++ --version  # Should show version with C++17 support
```

