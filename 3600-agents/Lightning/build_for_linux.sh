#!/bin/bash
# Script to prepare files for Linux compilation
# This creates a zip without macOS-specific files

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "Preparing CPPHikaru_3 for Linux compilation..."
echo "Creating zip file with source code..."

# Create zip excluding macOS binaries and build artifacts
zip -r CPPHikaru_3.zip . \
    -x "*.dylib" \
    -x "*.so" \
    -x "*.dll" \
    -x "*.o" \
    -x "build/*" \
    -x "*.a" \
    -x "__pycache__/*" \
    -x "*.pyc" \
    -x ".DS_Store" \
    -x "*.zip"

echo ""
echo "✓ Created CPPHikaru_3.zip"
echo ""
echo "To compile on Linux server:"
echo "  1. Upload CPPHikaru_3.zip to the server"
echo "  2. unzip CPPHikaru_3.zip"
echo "  3. cd CPPHikaru_3"
echo "  4. make"
echo ""
echo "The compiled libcpphikaru3.so will be ready to use!"

