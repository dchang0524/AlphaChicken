#!/bin/bash

# Script to create a minimal package for PACE cluster deployment
# This creates a tarball with only what's needed to run batch games

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_NAME="pace_package_$(date +%Y%m%d_%H%M%S)"
PACKAGE_DIR="${SCRIPT_DIR}/${PACKAGE_NAME}"

echo "Creating PACE package: ${PACKAGE_NAME}"
echo ""

# Create package directory structure
mkdir -p "${PACKAGE_DIR}"
mkdir -p "${PACKAGE_DIR}/3600-agents"
mkdir -p "${PACKAGE_DIR}/engine"

# Copy CPPHikaru_3 agent (source + compiled Linux .so)
echo "Copying CPPHikaru_3 agent..."
mkdir -p "${PACKAGE_DIR}/3600-agents/CPPHikaru_3"
cp -r "${SCRIPT_DIR}/3600-agents/CPPHikaru_3"/*.cpp "${PACKAGE_DIR}/3600-agents/CPPHikaru_3/"
cp -r "${SCRIPT_DIR}/3600-agents/CPPHikaru_3"/*.h "${PACKAGE_DIR}/3600-agents/CPPHikaru_3/"
cp -r "${SCRIPT_DIR}/3600-agents/CPPHikaru_3"/*.py "${PACKAGE_DIR}/3600-agents/CPPHikaru_3/"
cp -r "${SCRIPT_DIR}/3600-agents/CPPHikaru_3"/Makefile "${PACKAGE_DIR}/3600-agents/CPPHikaru_3/" 2>/dev/null || true
cp -r "${SCRIPT_DIR}/3600-agents/CPPHikaru_3"/CMakeLists.txt "${PACKAGE_DIR}/3600-agents/CPPHikaru_3/" 2>/dev/null || true
cp -r "${SCRIPT_DIR}/3600-agents/CPPHikaru_3"/build_linux_so.sh "${PACKAGE_DIR}/3600-agents/CPPHikaru_3/" 2>/dev/null || true

# Copy the Linux .so if it exists, otherwise they'll need to compile on PACE
if [ -f "${SCRIPT_DIR}/3600-agents/CPPHikaru_3/libcpphikaru3.so" ]; then
    cp "${SCRIPT_DIR}/3600-agents/CPPHikaru_3/libcpphikaru3.so" "${PACKAGE_DIR}/3600-agents/CPPHikaru_3/"
    echo "  ✓ Included compiled Linux library"
else
    echo "  ⚠ No Linux .so found - will need to compile on PACE"
fi

# Copy opponent agents (common opponents for testing)
echo "Copying opponent agents..."
OPPONENTS=("Hikaru_3" "Magnus_2" "Magnus" "Bobby_5")
for opponent in "${OPPONENTS[@]}"; do
    if [ -d "${SCRIPT_DIR}/3600-agents/${opponent}" ]; then
        cp -r "${SCRIPT_DIR}/3600-agents/${opponent}" "${PACKAGE_DIR}/3600-agents/"
        echo "  ✓ Included ${opponent}"
    fi
done

# Copy engine directory
echo "Copying game engine..."
cp -r "${SCRIPT_DIR}/engine"/* "${PACKAGE_DIR}/engine/"
echo "  ✓ Engine files copied"

# Create matches directory for game results
mkdir -p "${PACKAGE_DIR}/3600-agents/matches"
echo "  ✓ Created matches directory"

# Copy batch script
echo "Copying batch runner script..."
cp "${SCRIPT_DIR}/run_batch_games.sh" "${PACKAGE_DIR}/"
chmod +x "${PACKAGE_DIR}/run_batch_games.sh"
echo "  ✓ Batch script copied"

# Copy SLURM scripts if they exist
if [ -f "${SCRIPT_DIR}/run_batch_pace.slurm" ]; then
    cp "${SCRIPT_DIR}/run_batch_pace.slurm" "${PACKAGE_DIR}/"
    echo "  ✓ SLURM script copied"
fi
if [ -f "${SCRIPT_DIR}/run_batch_pace_array.slurm" ]; then
    cp "${SCRIPT_DIR}/run_batch_pace_array.slurm" "${PACKAGE_DIR}/"
    echo "  ✓ SLURM array script copied"
fi
if [ -f "${SCRIPT_DIR}/run_single_game.sh" ]; then
    cp "${SCRIPT_DIR}/run_single_game.sh" "${PACKAGE_DIR}/"
    chmod +x "${PACKAGE_DIR}/run_single_game.sh"
    echo "  ✓ Single game runner copied"
fi
if [ -f "${SCRIPT_DIR}/aggregate_results.sh" ]; then
    cp "${SCRIPT_DIR}/aggregate_results.sh" "${PACKAGE_DIR}/"
    chmod +x "${PACKAGE_DIR}/aggregate_results.sh"
    echo "  ✓ Results aggregator copied"
fi

# Copy requirements.txt
if [ -f "${SCRIPT_DIR}/requirements.txt" ]; then
    cp "${SCRIPT_DIR}/requirements.txt" "${PACKAGE_DIR}/"
    echo "  ✓ Requirements file copied"
fi

# Create setup script for PACE
cat > "${PACKAGE_DIR}/setup_on_pace.sh" << 'EOFSCRIPT'
#!/bin/bash
# Setup script to run on PACE cluster

set -e

echo "Setting up AlphaChicken on PACE..."

# Check if we need to compile the library
if [ ! -f "3600-agents/CPPHikaru_3/libcpphikaru3.so" ]; then
    echo "Compiling C++ library..."
    cd 3600-agents/CPPHikaru_3
    
    # On Linux (PACE), use Makefile directly - no Docker needed
    if [ -f "Makefile" ]; then
        echo "  Using Makefile to compile..."
        make clean
        make
        echo "  ✓ Library compiled successfully"
    else
        echo "ERROR: No Makefile found!"
        exit 1
    fi
    cd ../..
else
    echo "  ✓ Library already exists"
fi

# Set library path
export LD_LIBRARY_PATH="${PWD}/3600-agents/CPPHikaru_3:${LD_LIBRARY_PATH}"

echo ""
echo "Setup complete!"
echo ""
echo "Note: Make sure to use Python 3.10+ when running games:"
echo "  module load python/3.10"
echo ""
echo "To run batch games:"
echo "  ./run_batch_games.sh Hikaru_3 100"
echo ""
EOFSCRIPT

chmod +x "${PACKAGE_DIR}/setup_on_pace.sh"

# Create a README
cat > "${PACKAGE_DIR}/README_PACE.md" << 'EOF'
# PACE Deployment Package

## Quick Start

1. **Extract the package:**
   ```bash
   tar -xzf pace_package_*.tar.gz
   cd pace_package_*
   ```

2. **Run setup:**
   ```bash
   bash setup_on_pace.sh
   ```
   This will compile the C++ library if needed.

3. **Run batch games:**
   ```bash
   ./run_batch_games.sh Hikaru_3 100
   ```

## What's Included

- `CPPHikaru_3/` - Your C++ agent (source + compiled library if available)
- `Hikaru_3/` - Opponent agent for testing
- `engine/` - Game engine code
- `run_batch_games.sh` - Batch runner script
- `requirements.txt` - Python dependencies

## Dependencies

Make sure Python 3 and a C++ compiler (g++) are available on PACE.
The requirements.txt lists Python packages needed.

## Batch Testing

The batch script will:
- Run games in parallel
- Track search depth by move range
- Generate detailed statistics

Results are saved in `batch_results_TIMESTAMP/` directory.
EOF

# Create tarball (use ustar format to avoid macOS extended attributes)
echo ""
echo "Creating tarball..."
cd "${SCRIPT_DIR}"
tar --format=ustar -czf "${PACKAGE_NAME}.tar.gz" "${PACKAGE_NAME}/"
rm -rf "${PACKAGE_NAME}"

echo ""
echo "✓ Package created: ${PACKAGE_NAME}.tar.gz"
echo ""
echo "To upload to PACE:"
echo "  scp ${PACKAGE_NAME}.tar.gz your_username@login.pace.gatech.edu:~/"
echo ""
echo "Then on PACE:"
echo "  tar -xzf ${PACKAGE_NAME}.tar.gz"
echo "  ls -d pace_package_*  # Find the extracted directory"
echo "  cd pace_package_202*  # Use tab completion or copy exact name"
echo "  bash setup_on_pace.sh"
echo "  ./run_batch_games.sh Hikaru_3 100"

