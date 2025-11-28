
import sys
import os
import json
import random
import time
import copy
import pathlib

# Add engine to path
current_dir = os.path.dirname(os.path.abspath(__file__))
engine_dir = os.path.abspath(os.path.join(current_dir, "../../engine"))
sys.path.append(engine_dir)

from gameplay import play_game
from game.board import Board

# Weights to tune and their initial values/ranges
TUNABLE_WEIGHTS = [
    "W_SPACE_MIN", "W_SPACE_MAX",
    "W_MAT_MIN", "W_MAT_MAX",
    "W_FRAG",
    "W_FRONTIER_DIST",
    "FRONTIER_COEFF",
    "W_WEIGHTED_CONTESTED",
    "TRAP_WEIGHT",
    "BASE_EGG", "BASE_PLAIN", "BASE_TURD",
    "DIR_WEIGHT",
    "TURD_CENTER_BONUS", "TURD_FRONTIER_BONUS"
]

def get_current_weights():
    # Load from weights.py (by importing it)
    # We need to import it dynamically or just parse the file/json
    # Simplest is to read weights.json if exists, else read from weights.py defaults
    # But we can just start with a known dict for now or import the class
    sys.path.append(current_dir)
    from weights import HeuristicWeights
    weights = {}
    for name in TUNABLE_WEIGHTS:
        if hasattr(HeuristicWeights, name):
            weights[name] = getattr(HeuristicWeights, name)
    return weights

def save_weights(weights):
    with open(os.path.join(current_dir, "weights.json"), "w") as f:
        json.dump(weights, f, indent=4)

def perturb_weights(weights, noise_scale=0.1):
    new_weights = copy.deepcopy(weights)
    for k in new_weights:
        # Add gaussian noise proportional to value magnitude
        val = new_weights[k]
        if isinstance(val, (int, float)):
            # Avoid changing 0.0 too much if it's meant to be 0, but here most are non-zero
            # If val is 0, add small noise
            base = abs(val) if val != 0 else 1.0
            noise = random.gauss(0, noise_scale * base)
            new_weights[k] += noise
    return new_weights

def run_match(agent_name, opponent_name):
    # Run a game
    # We assume we are in 3600-agents/Bobby_6/
    # play_game expects play_directory to be the parent of agent folders
    play_dir = os.path.abspath(os.path.join(current_dir, ".."))
    
    # Suppress output
    # sys.stdout = open(os.devnull, 'w')
    # sys.stderr = open(os.devnull, 'w')
    
    try:
        final_board, _, _, _, _ = play_game(
            play_dir,
            play_dir,
            agent_name,
            opponent_name,
            display_game=False,
            delay=0.0,
            clear_screen=False,
            record=True,
            limit_resources=False
        )
    except Exception as e:
        # sys.stdout = sys.__stdout__
        # sys.stderr = sys.__stderr__
        print(f"Game failed: {e}")
        return -1 # Error
        
    # sys.stdout = sys.__stdout__
    # sys.stderr = sys.__stderr__
    
    # Determine winner
    # We are agent_name (Player A usually if passed first)
    # play_game returns final_board. 
    # We need to know if agent_name was player A or B.
    # In play_game, player_a_name is the first arg.
    
    # Check score
    eggs_a = final_board.chicken_player.eggs_laid
    eggs_b = final_board.chicken_enemy.eggs_laid
    
    # If we are player A
    score = eggs_a - eggs_b
    return score

def train(iterations=5):
    print("Starting training...")
    current_weights = get_current_weights()
    save_weights(current_weights) # Ensure json exists
    
    best_weights = current_weights
    
    # Baseline performance
    print("Evaluating baseline...")
    baseline_score = run_match("Bobby_6", "Bobby_5_2")
    print(f"Baseline Score: {baseline_score}")
    
    best_score = baseline_score
    
    for i in range(iterations):
        print(f"Iteration {i+1}/{iterations}")
        
        # Perturb
        candidate_weights = perturb_weights(best_weights, noise_scale=0.2)
        save_weights(candidate_weights)
        
        # Evaluate
        score = run_match("Bobby_6", "Bobby_5_2")
        print(f"  Candidate Score: {score}")
        
        if score > best_score:
            print("  New best found!")
            best_score = score
            best_weights = candidate_weights
        else:
            # Revert
            print("  Reverting.")
            save_weights(best_weights)
            
    print("Training complete.")
    print(f"Best Score: {best_score}")
    save_weights(best_weights)

if __name__ == "__main__":
    train(iterations=5)
