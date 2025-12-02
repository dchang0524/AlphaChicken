# PACE Quick Start Guide

## Step 1: Create Package on Your Mac

```bash
cd /Users/johnmermigkas/Desktop/AlphaChicken
./create_pace_package.sh
```

This creates: `pace_package_YYYYMMDD_HHMMSS.tar.gz`

## Step 2: Upload to PACE

```bash
# Replace 'imermigkas3' with your PACE username if different
scp pace_package_*.tar.gz imermigkas3@login.pace.gatech.edu:~/
```

## Step 3: On PACE - Extract and Setup

```bash
# SSH into PACE
ssh imermigkas3@login.pace.gatech.edu

# Extract the package (if not already extracted)
if ! ls -d pace_package_* 1> /dev/null 2>&1; then
    tar -xzf pace_package_*.tar.gz
fi

# Check what package directories are available
echo "Available package directories:"
ls -d pace_package_* 2>/dev/null || echo "No package directories found"
echo ""

# YOU NEED TO MANUALLY CD TO YOUR PACKAGE DIRECTORY
# Replace the directory name below with the one you see above
# Example: cd pace_package_20251201_161600
cd pace_package_20251201_161600  # <-- CHANGE THIS to your actual package directory name

# Compile the C++ library (if not already compiled)
cd 3600-agents/CPPHikaru_3
make clean
make
cd ../..

# Set library path
export LD_LIBRARY_PATH="${PWD}/3600-agents/CPPHikaru_3:${LD_LIBRARY_PATH}"

# Add to your .bashrc to persist (optional)
echo 'export LD_LIBRARY_PATH="${HOME}/pace_package_*/3600-agents/CPPHikaru_3:${LD_LIBRARY_PATH}"' >> ~/.bashrc
```

## Step 4: Install Python Dependencies

```bash
# IMPORTANT: Load Python 3.10+ FIRST (required for match/case syntax)
module load python/3.10

# Verify version BEFORE creating venv
python3 --version  # Should show 3.10 or higher

# Remove old virtual environment if it exists (if it was created with Python 3.9)
if [ -d ~/cpphikaru_env ]; then
    echo "Removing old virtual environment..."
    rm -rf ~/cpphikaru_env
fi

# Create virtual environment with Python 3.10+ (must be done AFTER loading module)
python3 -m venv ~/cpphikaru_env

# Activate and verify Python version inside venv
source ~/cpphikaru_env/bin/activate
python3 --version  # Should STILL show 3.10 or higher (not 3.9!)

# Install dependencies
pip install --upgrade pip
pip install numpy psutil
```

## Step 5: Test Single Game (Optional)

```bash
# Make sure you're in the package directory
cd ~/pace_package_*

# Load Python 3.10+ first
module load python/3.10
source ~/cpphikaru_env/bin/activate  # Activate venv if using it

# Test one game to make sure everything works
bash run_single_game.sh 1 Hikaru_3

# Check if it worked
ls -la batch_results/match_data/
ls -la 3600-agents/matches/  # Should see a match JSON file
```

## Step 6: Submit Batch Job Array

```bash
# Make sure you're in the package directory
cd ~/pace_package_*

# Submit 100 games as job array (uses --array=1-100)
# The SLURM script will automatically load Python 3.10+
sbatch run_batch_pace_array.slurm

# Or with custom opponent:
OPPONENT=Magnus sbatch run_batch_pace_array.slurm

# Or change number of games (edit --array=1-100 in the .slurm file, or override):
sbatch --array=1-50 run_batch_pace_array.slurm  # Only 50 games

# Check job status
squeue -u imermigkas3

# See specific job array
squeue -j JOB_ID  # Replace JOB_ID with your job number
```

This will:
- Submit 100 separate SLURM jobs (or however many you specify)
- Each job runs 1 game independently
- Jobs run in parallel across available nodes
- Much faster than running sequentially!
- Automatically loads Python 3.10+ (required for code)

## Step 7: Monitor Jobs

```bash
# See all your jobs
squeue -u imermigkas3

# See specific job array
squeue -j JOB_ID  # Replace JOB_ID with your job number

# Cancel all jobs in array (if needed)
scancel JOB_ID

# View output from a specific game (e.g., game 42)
cat cpphikaru-JOB_ID_42.out
cat cpphikaru-JOB_ID_42.err
```

## Step 8: Aggregate Results

After all jobs complete:

```bash
# Make sure you're in the package directory
cd ~/pace_package_*

# Aggregate all results
bash aggregate_results.sh
```

This prints:
- Win/Loss/Tie statistics
- Average egg differences
- Win reasons breakdown
- **Average search depth by move range (1-10, 11-20, 21-30, 31-40)**

## Troubleshooting

### Python version error (SyntaxError with match/case)
```bash
# Make sure you're using Python 3.10+
module load python/3.10
python3 --version  # Should show 3.10 or higher

# The SLURM scripts automatically load Python 3.10+, but if running manually:
module load python/3.10
source ~/cpphikaru_env/bin/activate
```

### Library not found error
```bash
export LD_LIBRARY_PATH="${PWD}/3600-agents/CPPHikaru_3:${LD_LIBRARY_PATH}"
```

### Python version reverts to 3.9 after activating venv
If `python3 --version` shows 3.9 after `source ~/cpphikaru_env/bin/activate`, your venv was created with Python 3.9:

```bash
# Deactivate current venv
deactivate

# Remove old venv
rm -rf ~/cpphikaru_env

# Load Python 3.10 FIRST (very important!)
module load python/3.10

# Verify: should show 3.10+
python3 --version

# NOW create venv with Python 3.10
python3 -m venv ~/cpphikaru_env

# Activate and verify
source ~/cpphikaru_env/bin/activate
python3 --version  # Should still be 3.10+ (not 3.9!)

# Reinstall dependencies
pip install numpy psutil
```

### Check if library compiled correctly
```bash
ls -lh 3600-agents/CPPHikaru_3/libcpphikaru3.so
file 3600-agents/CPPHikaru_3/libcpphikaru3.so  # Should say "ELF 64-bit"
```

### Jobs stuck in queue
- Check PACE status: `pace-status`
- Try reducing array size: `sbatch --array=1-10 run_batch_pace_array.slurm` (test with 10 first)

### Check individual game logs
```bash
# Check if a specific game finished
ls batch_results/match_data/game_42_match.json

# View depth logs
cat batch_results/depth_logs/game_42_depth.log
```

## Example: Full Workflow

```bash
# 1. On your Mac - create package
./create_pace_package.sh

# 2. Upload
scp pace_package_*.tar.gz imermigkas3@login.pace.gatech.edu:~/

# 3. On PACE - extract
ssh imermigkas3@login.pace.gatech.edu
tar -xzf pace_package_*.tar.gz
cd pace_package_*

# 4. Compile
cd 3600-agents/CPPHikaru_3 && make && cd ../..
export LD_LIBRARY_PATH="${PWD}/3600-agents/CPPHikaru_3:${LD_LIBRARY_PATH}"

# 5. Setup Python (use Python 3.10+)
module load python/3.10
python3 -m venv ~/cpphikaru_env
source ~/cpphikaru_env/bin/activate
pip install numpy psutil

# 6. Submit jobs
sbatch run_batch_pace_array.slurm

# 7. Wait for jobs to complete (check with squeue)

# 8. Get results
bash aggregate_results.sh
```

## Quick Reference

- **Package creation**: `./create_pace_package.sh` (on Mac)
- **Upload**: `scp pace_package_*.tar.gz user@login.pace.gatech.edu:~/`
- **Extract**: `tar -xzf pace_package_*.tar.gz && cd pace_package_*`
- **Compile**: `cd 3600-agents/CPPHikaru_3 && make`
- **Python**: `module load python/3.10` (required for code)
- **Submit jobs**: `sbatch run_batch_pace_array.slurm`
- **Check status**: `squeue -u imermigkas3`
- **Get results**: `bash aggregate_results.sh`

