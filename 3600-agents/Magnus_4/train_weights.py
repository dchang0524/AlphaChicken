"""Reinforcement learning tuner for Magnus heuristic weights.

This script uses an Evolution Strategies style update to refine the heuristic
weights in ``weights.py``. It repeatedly perturbs the current weights, runs a
batch of self-play games against configurable opponents, and nudges the weights
in the direction of perturbations that produced stronger win rates.

Example usage (run from repo root):
    python 3600-agents/Magnus/train_weights.py \
        --opponent Magnus Bobby_2 \
        --iterations 50 \
        --population 20 \
        --games-per-candidate 16 \
        --eval-games 32 \
        --output 3600-agents/Magnus/weights.json
"""

from __future__ import annotations

import argparse
import io
import os
import random
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path
from typing import Iterable, List

import numpy as np

# Ensure engine modules are importable
REPO_ROOT = Path(__file__).resolve().parents[2]
ENGINE_DIR = REPO_ROOT / "engine"
if str(ENGINE_DIR) not in sys.path:
    sys.path.append(str(ENGINE_DIR))

from gameplay import play_game  # type: ignore
from game.enums import ResultArbiter  # type: ignore
from weights import HeuristicWeights

PLAY_DIRECTORY = REPO_ROOT / "3600-agents"
ALLOWED_OPPONENTS = ["Magnus", "Bobby_2", "Bobby_5_2", "Bobby_5_3", "Bobby_3"]


def write_temp_weights(vector: Iterable[float]) -> Path:
    """Write a temporary weights JSON file for the Magnus agent."""

    fd, path = tempfile.mkstemp(prefix="magnus_weights_", suffix=".json")
    os.close(fd)
    HeuristicWeights.update_from_vector(list(vector))
    HeuristicWeights.save_to_file(path)
    return Path(path)


def run_single_game(weight_path: Path, opponent: str, play_as_a: bool) -> float:
    """Play one game and return a reward (+1 win, -1 loss, 0 tie)."""

    player_a = "Magnus" if play_as_a else opponent
    player_b = opponent if play_as_a else "Magnus"

    previous_env = os.environ.get("MAGNUS_WEIGHTS_PATH")
    os.environ["MAGNUS_WEIGHTS_PATH"] = str(weight_path)

    # Silence per-game prints to keep HPC logs concise
    buffer = io.StringIO()
    try:
        with redirect_stdout(buffer):
            board, *_ = play_game(
                str(PLAY_DIRECTORY),
                str(PLAY_DIRECTORY),
                player_a,
                player_b,
                display_game=False,
                delay=0.0,
                clear_screen=False,
                record=False,
                limit_resources=True,
            )
    finally:
        if previous_env is None:
            os.environ.pop("MAGNUS_WEIGHTS_PATH", None)
        else:
            os.environ["MAGNUS_WEIGHTS_PATH"] = previous_env

    winner = board.get_winner()
    if winner == ResultArbiter.TIE:
        return 0.0
    if (winner == ResultArbiter.PLAYER_A and play_as_a) or (
        winner == ResultArbiter.PLAYER_B and not play_as_a
    ):
        return 1.0
    return -1.0


def evaluate_vector(vector: np.ndarray, opponents: List[str], games_per_opponent: int) -> float:
    """Average reward across opponents and game sides."""

    weight_path = write_temp_weights(vector)
    try:
        total_reward = 0.0
        total_games = 0
        for opponent in opponents:
            for game_idx in range(games_per_opponent):
                reward = run_single_game(weight_path, opponent, play_as_a=(game_idx % 2 == 0))
                total_reward += reward
                total_games += 1
        return total_reward / max(1, total_games)
    finally:
        weight_path.unlink(missing_ok=True)


def evolution_strategies(
    iterations: int,
    population: int,
    sigma: float,
    lr: float,
    games_per_candidate: int,
    eval_games: int,
    opponents: List[str],
    output_path: Path,
) -> None:
    base_vector = np.asarray(HeuristicWeights.to_vector(), dtype=float)
    best_vector = base_vector.copy()
    best_score = evaluate_vector(best_vector, opponents, eval_games)

    print(f"Initial score: {best_score:.3f} with weights saved to {output_path}")

    for iteration in range(1, iterations + 1):
        noises = np.random.randn(population, base_vector.size)
        rewards = []

        for idx, noise in enumerate(noises):
            candidate = base_vector + sigma * noise
            reward = evaluate_vector(candidate, opponents, games_per_candidate)
            rewards.append(reward)
            print(
                f"Iter {iteration:03d} cand {idx:02d}: reward={reward:.3f}"
            )

        reward_array = np.asarray(rewards, dtype=float)
        standardized = (reward_array - reward_array.mean()) / (reward_array.std() + 1e-8)
        gradient = standardized @ noises / (population * sigma)
        base_vector += lr * gradient

        eval_score = evaluate_vector(base_vector, opponents, eval_games)
        print(
            f"Iter {iteration:03d} eval: score={eval_score:.3f}, mean_cand={reward_array.mean():.3f}"
        )

        if eval_score > best_score:
            best_score = eval_score
            best_vector = base_vector.copy()
            HeuristicWeights.update_from_vector(best_vector.tolist())
            HeuristicWeights.save_to_file(output_path)
            print(f"  Updated best score -> {best_score:.3f}, saved weights.")

    HeuristicWeights.update_from_vector(best_vector.tolist())
    HeuristicWeights.save_to_file(output_path)
    print(f"Training finished. Best score: {best_score:.3f}. Weights saved to {output_path}.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Magnus heuristic weights via RL.")
    parser.add_argument(
        "--opponent",
        dest="opponents",
        nargs="+",
        default=["Magnus"],
        choices=ALLOWED_OPPONENTS,
        help=(
            "One or more opponents to train against (allowed: Magnus self-play, "
            "Bobby_2, Bobby_5_2, Bobby_5_3, Bobby_3)."
        ),
    )
    parser.add_argument("--iterations", type=int, default=10, help="Training iterations")
    parser.add_argument("--population", type=int, default=8, help="Perturbations per iteration")
    parser.add_argument("--sigma", type=float, default=0.1, help="Gaussian noise scale")
    parser.add_argument("--lr", type=float, default=0.05, help="Learning rate for updates")
    parser.add_argument(
        "--games-per-candidate",
        type=int,
        default=8,
        help="Number of games each perturbation is evaluated on (split across sides).",
    )
    parser.add_argument(
        "--eval-games",
        type=int,
        default=16,
        help="Number of games to score the current policy each iteration.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=HeuristicWeights._default_weights_path(),
        help="Destination JSON file for the tuned weights.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    random.seed(0)
    np.random.seed(0)
    evolution_strategies(
        iterations=args.iterations,
        population=args.population,
        sigma=args.sigma,
        lr=args.lr,
        games_per_candidate=args.games_per_candidate,
        eval_games=args.eval_games,
        opponents=args.opponents,
        output_path=args.output,
    )
