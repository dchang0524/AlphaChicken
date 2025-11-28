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

# Weights to tune
TUNABLE_WEIGHTS = [
    "W_SPACE_MIN", "W_SPACE_MAX",
    "W_MAT_MIN", "W_MAT_MAX",
    "W_FRAG",
    "W_FRONTIER_DIST",
    "FRONTIER_COEFF",
    "W_WEIGHTED_CONTESTED",
    # TRAP_WEIGHT is not in evaluate gradient
]

def get_current_weights():
    """
    Load current weights from weights.py (HeuristicWeights class).
    Only pull the ones we actually want to train.
    """
    sys.path.append(current_dir)
    from weights import HeuristicWeights
    weights = {}
    for name in TUNABLE_WEIGHTS:
        if hasattr(HeuristicWeights, name):
            weights[name] = getattr(HeuristicWeights, name)
    return weights

def save_weights(weights):
    """
    Save weights to weights.json so the agent can load them.
    """
    with open(os.path.join(current_dir, "weights.json"), "w") as f:
        json.dump(weights, f, indent=4)

def run_self_play_match():
    """
    Run a self-play game: Bobby_6 vs Bobby_6.

    Assumes that Bobby_6 is instrumented to:
    - read weights.json at startup
    - log per-move gradients to game_trace.jsonl in current_dir

    Returns:
        score_a: egg difference (A - B) at game end
        trace_path: path to the trace file with per-move gradients
    """
    play_dir = os.path.abspath(os.path.join(current_dir, ".."))
    
    # Clear previous trace
    trace_path = os.path.join(current_dir, "game_trace.jsonl")
    if os.path.exists(trace_path):
        os.remove(trace_path)
    
    try:
        final_board, _, _, _, _ = play_game(
            play_dir,
            play_dir,
            "Bobby_6",   # Player A
            "Bobby_6",   # Player B
            display_game=False,
            delay=0.0,
            clear_screen=False,
            record=True,
            limit_resources=False
        )
    except Exception as e:
        print(f"Game failed: {e}")
        return None, None

    # Determine winner from final board POV (Player A vs Player B)
    eggs_a = final_board.chicken_player.eggs_laid
    eggs_b = final_board.chicken_enemy.eggs_laid
    
    score_a = eggs_a - eggs_b
    return score_a, trace_path

def train_td(episodes=100, alpha=0.01, gamma=1.0):
    """
    Monte Carlo value learning on self-play games.

    Assumptions:
    - For each move, your agent logs:
        {
          "turn": <int>,
          "grads": { weight_name: dV/dw }
        }
      into game_trace.jsonl.
    - V(s) is linear in the tunable weights:
        V(s) = sum_k w_k * feature_k(s),
      and grads[k] == feature_k(s).
    - Evaluation is POV-relative to the side to move when the state was evaluated.

    Update rule (per visited state s):
        target = final game outcome (+1/-1/0 from that player's POV)
        V_est  = current V(s) under weights
        td_err = target - V_est
        w_k   += alpha * td_err * grad_k
    """
    print("Starting TD Learning (Self-Play)...")
    weights = get_current_weights()
    save_weights(weights)
    
    for i in range(episodes):
        print(f"Episode {i+1}/{episodes}")
        
        score_a, trace_path = run_self_play_match()
        
        if score_a is None:
            print("  Game failed, skipping episode.")
            continue
            
        print(f"  Game finished. Score A (eggs_a - eggs_b): {score_a}")
        
        # Reward from Player A's POV
        if score_a > 0:
            reward_a = 1.0
        elif score_a < 0:
            reward_a = -1.0
        else:
            reward_a = 0.0
        
        reward_b = -reward_a  # zero-sum assumption
        
        if not os.path.exists(trace_path):
            print("  No trace file found.")
            continue
            
        with open(trace_path, "r") as f:
            lines = f.readlines()
            
        # Accumulate weight updates across all states in the game
        updates = {k: 0.0 for k in weights}
        count_a = 0
        count_b = 0
        
        for line in lines:
            try:
                entry = json.loads(line)
            except Exception:
                continue

            if "turn" not in entry or "grads" not in entry:
                continue

            turn = entry["turn"]
            grads = entry["grads"]

            # Infer which player made this move from turn parity:
            # Assuming turn_count starts at 0 and:
            #   A moves: turns 0, 2, 4, ...
            #   B moves: turns 1, 3, 5, ...
            is_player_a = (turn % 2 == 0)
            target = reward_a if is_player_a else reward_b

            # Build V_est(s) from current weights and grads:
            # V_est = sum_k w_k * grad_k, for tunable weights present in grads
            V_est = 0.0
            for k, w_val in weights.items():
                g_val = grads.get(k, None)
                if g_val is None:
                    continue
                V_est += w_val * g_val

            # TD/MC error for this state
            td_error = target - V_est

            # Gradient step: w_k += alpha * td_error * grad_k
            for k, g in grads.items():
                if k in updates:
                    updates[k] += alpha * td_error * g

            if is_player_a:
                count_a += 1
            else:
                count_b += 1
        
        print(f"  Processed {count_a} moves for A, {count_b} moves for B.")

        # Apply accumulated updates
        for k in weights:
            weights[k] += updates[k]
        
        save_weights(weights)
        print("  Weights updated.")

if __name__ == "__main__":
    train_td(episodes=10)
