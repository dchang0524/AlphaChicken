
import json
import os
from pathlib import Path
from typing import Dict, List


class HeuristicWeights:
    """Container for Magnus heuristic weights.

    The attributes below define the current weights used by the agent. They can
    be tuned via reinforcement learning by writing a JSON file and setting the
    ``MAGNUS_WEIGHTS_PATH`` environment variable (or by writing to the default
    ``weights.json`` path next to this module).
    """

    # --- Weights in egg units (tunable) ---

    # 1. Base Egg Difference
    # Implicitly 1.0 (score is in egg units)

    # 2. Voronoi Space Difference
    # Weight for reachable unclaimed eggs in my voronoi region
    W_VORONOI_EGG = 1.5 # changed form 0.8

    # 3. Trapdoor Risk
    # Weight for probability of stepping on trapdoor
    # Risk = W_TRAP_RISK * prob * penalty
    W_TRAP_RISK = 1.0

    # Constant for total trapdoor term calculation
    # term = W_TRAP_CONST * (4 + 8 * openness * ...)
    W_TRAP_CONST = 0.5
    TRAP_WEIGHT = 1.0

    # 4. Closeness to Egg
    # Bonus for being close to an egg if min_dist >= depth/2
    W_EGG_DIST_PENALTY = 0.5

    # 5. Blocks (Safe Eggs)
    # Weight for unclaimed eggs where opponent can't reach
    W_SAFE_EGG = 1.0

    # 6. Turd Bonus (Savings)
    # Bonus per turd left
    W_TURD_SAVINGS = 0.3

    # 7. Turd Penalty (Bad Placement)
    # Penalty for wasting a turd
    W_BAD_TURD = 0.0

    # 8. Entropy Bonus
    W_ENTROPY = 0.1

    # 9/10. Contested Squares Significance
    # Penalty = significance * W_CONTESTED_SIG * distance * (1 - progression)
    W_CONTESTED_SIG = 1.0
    CONTESTED_OPENNESS_CORRELATION = 1.0

    # 11. Dynamic Weights / Openness
    # Openness normalization factor
    OPENNESS_FACTOR = 8.0

    # 12. Loss Prevention
    # Penalty for lost positions
    W_LOSS_PENALTY = 100.0

    # Move Ordering Weights (Keep these)
    BASE_EGG = 3.0
    BASE_PLAIN = 2.0
    BASE_TURD = 1.0
    DIR_WEIGHT = 0.5
    TURD_CENTER_BONUS = 2.0
    TURD_FRONTIER_BONUS = 3.0

    TRAINABLE_KEYS: List[str] = [
        "W_VORONOI_EGG",
        "W_TRAP_RISK",
        "W_TRAP_CONST",
        "TRAP_WEIGHT",
        "W_EGG_DIST_PENALTY",
        "W_SAFE_EGG",
        "W_TURD_SAVINGS",
        "W_BAD_TURD",
        "W_ENTROPY",
        "W_CONTESTED_SIG",
        "CONTESTED_OPENNESS_CORRELATION",
        "OPENNESS_FACTOR",
        "W_LOSS_PENALTY",
    ]

    @classmethod
    def _default_weights_path(cls) -> Path:
        return Path(__file__).with_name("weights.json")

    @classmethod
    def to_dict(cls) -> Dict[str, float]:
        return {key: getattr(cls, key) for key in cls.TRAINABLE_KEYS}

    @classmethod
    def update_from_dict(cls, data: Dict[str, float]) -> None:
        for key, value in data.items():
            if hasattr(cls, key):
                setattr(cls, key, float(value))

    @classmethod
    def to_vector(cls) -> List[float]:
        return [float(getattr(cls, key)) for key in cls.TRAINABLE_KEYS]

    @classmethod
    def update_from_vector(cls, values: List[float]) -> None:
        for key, value in zip(cls.TRAINABLE_KEYS, values):
            setattr(cls, key, float(value))

    @classmethod
    def load_from_file(cls, filepath: os.PathLike | str | None = None) -> None:
        path = Path(filepath) if filepath else cls._default_weights_path()
        if not path.exists():
            return
        try:
            with path.open("r") as f:
                data = json.load(f)
            cls.update_from_dict(data)
        except Exception as exc:  # pragma: no cover - defensive
            print(f"Error loading weights from {path}: {exc}")

    @classmethod
    def save_to_file(cls, filepath: os.PathLike | str | None = None) -> None:
        path = Path(filepath) if filepath else cls._default_weights_path()
        try:
            with path.open("w") as f:
                json.dump(cls.to_dict(), f, indent=2)
        except Exception as exc:  # pragma: no cover - defensive
            print(f"Error saving weights to {path}: {exc}")


def _load_weights_on_import() -> None:
    env_path = os.getenv("MAGNUS_WEIGHTS_PATH")
    if env_path:
        HeuristicWeights.load_from_file(env_path)
    else:
        HeuristicWeights.load_from_file()


_load_weights_on_import()
