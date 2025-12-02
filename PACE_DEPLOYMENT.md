# PACE Cluster Deployment Guide

## Quick Answer

**You don't need to upload the entire AlphaChicken directory!** Use the packaging script to create a minimal package.

## Step 1: Create Package

Run on your local machine:

```bash
cd /Users/johnmermigkas/Desktop/AlphaChicken
./create_pace_package.sh
```

This creates `pace_package_YYYYMMDD_HHMMSS.tar.gz` with only what's needed.

## Step 2: Upload to PACE

```bash
scp pace_package_*.tar.gz your_username@login.pace.gatech.edu:~/
```

## Step 3: On PACE - Extract and Setup

```bash
# SSH into PACE
ssh your_username@login.pace.gatech.edu

# Extract package (warnings about extended headers are harmless)
tar -xzf pace_package_*.tar.gz

# List extracted directories and enter the right one
ls -d pace_package_*
cd pace_package_202*  # Use tab completion or list first

# OR if there's only one:
PACKAGE_DIR=$(ls -d pace_package_* | head -1)
cd "$PACKAGE_DIR"

# Run setup (compiles library if needed)
bash setup_on_pace.sh
```

## Step 4: Submit Batch Job

### Option A: Using SLURM Job Array (RECOMMENDED - True Parallel Execution)

```bash
# Submit 100 jobs, each running one game in parallel
sbatch run_batch_pace_array.slurm

# Or with custom opponent:
OPPONENT=Hikaru_3 sbatch run_batch_pace_array.slurm
```

This will:
- Submit 100 **separate SLURM jobs** (true parallelism across nodes)
- Each job runs 1 game independently
- Can use many nodes simultaneously (much faster!)
- Results saved to `batch_results/`

After all jobs complete, aggregate results:
```bash
bash aggregate_results.sh
```

### Option B: Single-node parallel (slower, uses Python multiprocessing)

```bash
sbatch run_batch_pace.slurm Hikaru_3 100
```

This will:
- Request 16 CPU cores on one node
- Run games using Python multiprocessing
- Much slower (sequential on one machine)
- Only use this if job arrays aren't available

### Option B: Direct execution (for testing)

```bash
# Make sure library path is set
export LD_LIBRARY_PATH="${PWD}/3600-agents/CPPHikaru_3:${LD_LIBRARY_PATH}"

# Run batch games
./run_batch_games.sh Hikaru_3 100
```

## What Gets Packaged

The package includes:
- ✅ `CPPHikaru_3/` - Your agent (source + compiled `.so` if available)
- ✅ `Hikaru_3/` - Opponent agent
- ✅ `engine/` - Game engine
- ✅ `run_batch_games.sh` - Batch runner
- ✅ `run_batch_pace.slurm` - SLURM job script
- ✅ `requirements.txt` - Python dependencies
- ✅ Setup scripts

It excludes:
- ❌ Other agents (unless needed as opponents)
- ❌ Build artifacts (`*.o`, `build/`, etc.)
- ❌ Match history
- ❌ Documentation files

## Customizing the Batch Job

Edit `run_batch_pace.slurm` to change:
- Number of CPU cores: `--cpus-per-task=16`
- Memory: `--mem=32GB`
- Wall time: `--time=12:00:00`
- Number of games: Pass as argument or edit default

## Checking Results

After the job completes:

```bash
# Check job status
squeue -u your_username

# View output
cat cpphikaru-batch-JOBID.out

# Check results
ls -la batch_results_*/
```

## Troubleshooting

### Library not found
```bash
export LD_LIBRARY_PATH="${PWD}/3600-agents/CPPHikaru_3:${LD_LIBRARY_PATH}"
```

### Compilation fails
- Make sure `g++` is available: `module load gcc`
- Check that all `.cpp` and `.h` files are included

### Python import errors
- Install dependencies: `pip install -r requirements.txt`
- Make sure you're using Python 3.10: `module load python/3.10`

### Batch script not found
- Make sure you're in the extracted package directory
- Check that `run_batch_games.sh` is executable: `chmod +x run_batch_games.sh`

