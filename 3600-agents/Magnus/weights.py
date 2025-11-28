
class HeuristicWeights:
    # --- Weights in egg units (tunable) ---

    # 1. Base Egg Difference
    # Implicitly 1.0 (score is in egg units)

    # 2. Voronoi Space Difference
    # Weight for reachable unclaimed eggs in my voronoi region
    W_VORONOI_EGG = 0.7

    # 3. Trapdoor Risk
    # Weight for probability of stepping on trapdoor
    # Risk = W_TRAP_RISK * prob * penalty
    W_TRAP_RISK = 1.0 
    
    # Constant for total trapdoor term calculation
    # term = W_TRAP_CONST * (4 + 8 * openness * ...)
    W_TRAP_CONST = 1.0

    # 4. Closeness to Egg
    # Bonus for being close to an egg if min_dist >= depth/2
    W_EGG_DIST_PENALTY = 0.5

    # 5. Blocks (Safe Eggs)
    # Weight for unclaimed eggs where opponent can't reach
    W_SAFE_EGG = 0.9

    # 6. Turd Bonus (Savings)
    # Bonus per turd left
    W_TURD_SAVINGS = 0.5

    # 7. Turd Penalty (Bad Placement)
    # Penalty for wasting a turd
    W_BAD_TURD = 1.0

    # 8. Entropy Bonus
    W_ENTROPY = 0.1

    # 9/10. Contested Squares Significance
    # Penalty = significance * W_CONTESTED_SIG * distance * (1 - progression)
    W_CONTESTED_SIG = 1.0

    # 11. Dynamic Weights / Openness
    # Openness normalization factor
    OPENNESS_FACTOR = 8.0

    # 12. Loss Prevention
    # Penalty for lost positions
    W_LOSS_PENALTY = 100.0

    # Move Ordering Weights (Keep these)
    BASE_EGG   = 3.0
    BASE_PLAIN = 2.0
    BASE_TURD  = 1.0
    DIR_WEIGHT = 0.5
    TURD_CENTER_BONUS   = 2.0
    TURD_FRONTIER_BONUS = 3.0

    # Agent Weights
    TRAP_WEIGHT = 100.0 # Used in agent.py for path risk

#     @classmethod
#     def load_from_file(cls, filepath):
#         import json
#         import os
#         if not os.path.exists(filepath):
#             return
#         try:
#             with open(filepath, 'r') as f:
#                 data = json.load(f)
#             for key, value in data.items():
#                 if hasattr(cls, key):
#                     setattr(cls, key, value)
#         except Exception as e:
#             print(f"Error loading weights: {e}")

# import os
# # Load weights automatically when module is imported
# _current_dir = os.path.dirname(os.path.abspath(__file__))
# _weights_path = os.path.join(_current_dir, "weights.json")
# HeuristicWeights.load_from_file(_weights_path)
