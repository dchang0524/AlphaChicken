# Instructions for Zipping and Deploying

## For Deployment to Server

### Option 1: Compile Locally (if same OS as server)

If your server is the same OS as your local machine:

```bash
# Build the library
./build_simple.sh   # or ./build.sh if you have CMake

# Create zip (excluding build artifacts)
zip -r CPPHikaru_3.zip . -x "*.o" "build/*" "*.a" "__pycache__/*" "*.pyc"
```

Then upload `CPPHikaru_3.zip` to the server.

### Option 2: Compile on Server (recommended for different OS)

If your server is a different OS (e.g., Linux server, macOS local):

```bash
# Create zip with source code (no compiled library)
zip -r CPPHikaru_3.zip . \
    -x "*.so" "*.dylib" "*.dll" \
    -x "*.o" "build/*" "*.a" \
    -x "__pycache__/*" "*.pyc"
```

Then on the server:
```bash
unzip CPPHikaru_3.zip
cd CPPHikaru_3
./build_simple.sh   # or make, if you prefer
```

### What to Include in the Zip

**Always include:**
- All `.h` and `.cpp` files (source code)
- `agent.py` (Python wrapper)
- `CMakeLists.txt` or `Makefile` (build configuration)
- `README.md`, `BUILD_AND_TEST.md` (documentation)

**Include if compiled locally (same OS):**
- `libcpphikaru3.so` (Linux)
- `libcpphikaru3.dylib` (macOS)
- `cpphikaru3.dll` (Windows)

**Don't include:**
- `build/` directory (temporary build files)
- `*.o` files (object files)
- `__pycache__/` (Python cache)

## Quick Test Before Zipping

```bash
# Test that the library loads
cd 3600-agents/CPPHikaru_3
python3 -c "import agent; print('Import successful!')"
```

If this works, you're good to zip and deploy!

