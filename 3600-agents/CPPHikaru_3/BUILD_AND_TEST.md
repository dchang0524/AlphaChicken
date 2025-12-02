# Building and Testing CPPHikaru_3

## Step 1: Build the C++ Library

```bash
cd 3600-agents/CPPHikaru_3
./build.sh
```

This will:
- Create a `build/` directory
- Compile the C++ code into a shared library
- Copy the library (`libcpphikaru3.so` on Linux, `libcpphikaru3.dylib` on macOS) to the CPPHikaru_3 directory

**Alternative manual build:**
```bash
cd 3600-agents/CPPHikaru_3
mkdir -p build
cd build
cmake ..
make
# Then copy the library back:
cp libcpphikaru3.* ..  # (adjust for your platform)
```

## Step 2: Verify the Library Exists

After building, you should see one of these files in `3600-agents/CPPHikaru_3/`:
- `libcpphikaru3.so` (Linux)
- `libcpphikaru3.dylib` (macOS)
- `cpphikaru3.dll` (Windows)

## Step 3: Test the Agent

### Option 1: Test against another agent

From the project root:

```bash
cd engine
python3 run_local_agents.py CPPHikaru_3 Greedy
```

This will run CPPHikaru_3 vs Greedy. You can replace `Greedy` with any other agent name (e.g., `Hikaru_3`, `Bobby`, etc.).

### Option 2: Test against itself

```bash
cd engine
python3 run_local_agents.py CPPHikaru_3 CPPHikaru_3
```

### Option 3: Test with display

The `run_local_agents.py` script already has `display_game=True`, so you'll see the game board.

## Troubleshooting

### Library not found error

If you get an error like `ImportError: Could not find cpphikaru3 shared library`:

1. Make sure you built the library (Step 1)
2. Check that the library file exists in `3600-agents/CPPHikaru_3/`
3. On macOS, you might need to set the library path:
   ```bash
   export DYLD_LIBRARY_PATH=3600-agents/CPPHikaru_3:$DYLD_LIBRARY_PATH
   ```
4. On Linux, you might need:
   ```bash
   export LD_LIBRARY_PATH=3600-agents/CPPHikaru_3:$LD_LIBRARY_PATH
   ```

### Build errors

- Make sure you have CMake installed: `cmake --version`
- Make sure you have a C++17 compiler: `g++ --version` or `clang++ --version`
- If you get "pybind11 not found", that's fine - we're using ctypes now, not pybind11

### Import errors in Python

- Make sure you're running from the project root or have the right Python path
- The `agent.py` file should be in `3600-agents/CPPHikaru_3/`
- The library should be in the same directory as `agent.py`

### Runtime errors

- Check that all dependencies are installed
- Make sure the game engine files are accessible
- Check that the Board class interface matches what the agent expects

