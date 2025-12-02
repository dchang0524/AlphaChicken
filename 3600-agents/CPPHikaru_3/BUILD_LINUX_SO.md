# Building Linux .so File on macOS

## Method 1: Using Docker (Recommended)

### Prerequisites
1. Install Docker Desktop for Mac: https://www.docker.com/products/docker-desktop
2. Start Docker Desktop and wait for it to fully start (whale icon in menu bar should be steady)

### Steps
```bash
./build_linux_so.sh
./create_final_zip.sh
```

This will create `CPPHikaru_3_final.zip` with the Linux `.so` file ready to upload.

## Method 2: Manual Docker Build

If the script doesn't work, you can build manually:

```bash
# Build the Docker image
docker build -t cpphikaru3-builder .

# Create a container
CONTAINER_ID=$(docker create cpphikaru3-builder)

# Copy the .so file out
docker cp $CONTAINER_ID:/build/libcpphikaru3.so .

# Clean up
docker rm $CONTAINER_ID

# Verify it's a Linux library
file libcpphikaru3.so
```

## Troubleshooting

### "Docker daemon is not running"
- Open Docker Desktop
- Wait until the whale icon in the menu bar stops animating
- Try again

### "Cannot connect to Docker"
- Make sure Docker Desktop is fully started
- Try restarting Docker Desktop
- Check: `docker ps` should work without errors

### Alternative: Ask someone with Linux access
If Docker doesn't work, you can:
1. Use `build_for_linux.sh` to create a source-only zip
2. Have someone with Linux compile it: `make`
3. Get the `libcpphikaru3.so` back and include it in your final zip

