#!/bin/bash
# Build Linux .so file using Docker (for cross-compilation from macOS)

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "Building Linux .so file using Docker..."
echo ""

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed or not in PATH"
    echo ""
    echo "Please install Docker Desktop for Mac:"
    echo "  https://www.docker.com/products/docker-desktop"
    exit 1
fi

# Check if Docker daemon is running
if ! docker info > /dev/null 2>&1; then
    echo "Error: Docker daemon is not running"
    echo ""
    echo "Please start Docker Desktop and wait for it to fully start, then try again."
    exit 1
fi

# Build using Docker with x86_64 platform (for tournament server)
echo "Building Docker image for x86_64..."
if ! docker build --platform linux/amd64 -t cpphikaru3-builder .; then
    echo "Error: Docker build failed"
    exit 1
fi

# Extract the .so file from the container
echo "Extracting libcpphikaru3.so..."
CONTAINER_ID=$(docker create --platform linux/amd64 cpphikaru3-builder)
if [ -z "$CONTAINER_ID" ]; then
    echo "Error: Failed to create container"
    exit 1
fi

docker cp "$CONTAINER_ID:/build/libcpphikaru3.so" . || {
    echo "Error: Failed to copy .so file from container"
    docker rm "$CONTAINER_ID" > /dev/null 2>&1
    exit 1
}

docker rm "$CONTAINER_ID" > /dev/null 2>&1

# Verify it's a Linux library
if [ -f "libcpphikaru3.so" ]; then
    echo "✓ Successfully created libcpphikaru3.so"
    file libcpphikaru3.so
    ls -lh libcpphikaru3.so
    echo ""
    echo "The .so file is ready to include in your zip!"
else
    echo "Error: Failed to create libcpphikaru3.so"
    exit 1
fi

