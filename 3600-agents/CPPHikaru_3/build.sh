#!/bin/bash
# Build script for CPPHikaru_3

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "Building CPPHikaru_3..."

# Create build directory
mkdir -p build
cd build

# Run CMake
cmake ..

# Build
make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

# Copy library to parent directory
if [ -f "libcpphikaru3.so" ]; then
    cp libcpphikaru3.so ..
    echo "Build successful! Library copied to: $(cd .. && pwd)/libcpphikaru3.so"
elif [ -f "libcpphikaru3.dylib" ]; then
    cp libcpphikaru3.dylib ..
    echo "Build successful! Library copied to: $(cd .. && pwd)/libcpphikaru3.dylib"
elif [ -f "cpphikaru3.dll" ]; then
    cp cpphikaru3.dll ..
    echo "Build successful! Library copied to: $(cd .. && pwd)/cpphikaru3.dll"
else
    echo "Build completed but library file not found in expected location."
    echo "Please check the build directory for the output library."
    exit 1
fi

