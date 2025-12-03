#!/bin/bash
# Create final zip with Linux .so file included

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check if .so file exists
if [ ! -f "libcpphikaru3.so" ]; then
    echo "Error: libcpphikaru3.so not found!"
    echo "Please run ./build_linux_so.sh first to create it."
    exit 1
fi

# Verify it's a Linux library
file libcpphikaru3.so | grep -q "ELF.*shared object" || {
    echo "Warning: libcpphikaru3.so doesn't appear to be a Linux library"
    echo "File type:"
    file libcpphikaru3.so
}

echo "Creating final zip with Linux .so file..."
zip -r CPPHikaru_3_final.zip . \
    -x "*.dylib" \
    -x "*.dll" \
    -x "*.o" \
    -x "build/*" \
    -x "*.a" \
    -x "__pycache__/*" \
    -x "*.pyc" \
    -x ".DS_Store" \
    -x "*.zip" \
    -x "Dockerfile" \
    -x ".dockerignore" \
    -x "build_*.sh" \
    -x "LINUX_BUILD_INSTRUCTIONS.md"

echo ""
echo "✓ Created CPPHikaru_3_final.zip with libcpphikaru3.so"
echo "This zip is ready to upload to your server!"

