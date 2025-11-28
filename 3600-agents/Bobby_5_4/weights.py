
class HeuristicWeights:
    # --- Weights in egg units (tunable) ---

    # Space: important when open, but never completely zero.
    W_SPACE_MIN = 5.0   # closed
    W_SPACE_MAX = 25.0  # very open

    # Material: always matters, but ramps up hard toward the end.
    W_MAT_MIN   = 5.0   # early
    W_MAT_MAX   = 25.0  # late

    # Fragmentation: how bad it is if contested squares are spatially split.
    # Assumes vor.frag_score ∈ [0,1]
    W_FRAG      = 3.0

    # Frontier distance: how bad it is if I'm far from my most distant frontier.
    # Assumes vor.max_contested_dist = max dist(from my chicken, to any contested square)
    W_FRONTIER_DIST = 1.5

    # Frontier closeness bonus.
    FRONTIER_COEFF = 0.5

    # Weighted contested term
    # Reward holding contested squares that lead to large regions
    W_WEIGHTED_CONTESTED = 0.1

    # Panic threshold for changing behavior
    PANIC_THRESHOLD = 8

    # Move Ordering Weights
    BASE_EGG   = 3.0
    BASE_PLAIN = 2.0
    BASE_TURD  = 1.0

    # How much contested counts by direction matter
    DIR_WEIGHT = 0.5

    # TURD buffs
    TURD_CENTER_BONUS   = 2.0   # TURD > PLAIN if we’re closer to center
    TURD_FRONTIER_BONUS = 3.0   # TURD can rival/beat EGG near frontier

    # Agent Weights
    # Scale for path-risk penalty (per square)
    # delta_risk = -TRAP_WEIGHT * P(trap at square)
    TRAP_WEIGHT = 100.0
