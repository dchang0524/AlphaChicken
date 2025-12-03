#!/bin/bash
# Simple build script using Make (no CMake required)

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "Building CPPHikaru_3 using Make..."

# Check for g++ or clang++
if command -v g++ &> /dev/null; then
    CXX=g++
elif command -v clang++ &> /dev/null; then
    CXX=clang++
    # Use clang++ in Makefile
    sed -i.bak 's/CXX = g++/CXX = clang++/g' Makefile 2>/dev/null || \
    sed -i '' 's/CXX = g++/CXX = clang++/g' Makefile
else
    echo "Error: No C++ compiler found. Please install g++ or clang++"
    exit 1
fi

# Build using Make
make clean || true
make

# Detect OS and set library name
UNAME_S=$(uname -s)
if [ "$UNAME_S" = "Linux" ]; then
    LIBNAME="libcpphikaru3.so"
elif [ "$UNAME_S" = "Darwin" ]; then
    LIBNAME="libcpphikaru3.dylib"
else
    LIBNAME="cpphikaru3.dll"
fi

if [ -f "$LIBNAME" ]; then
    echo "Build successful! Library: $LIBNAME"
    echo "You can now zip this directory and send it to the server."
    echo ""
    echo "Note: If the server is a different OS (e.g., Linux vs macOS),"
    echo "you'll need to compile on the server or include source files."
else
    echo "Build failed - library not found"
    exit 1
fi

