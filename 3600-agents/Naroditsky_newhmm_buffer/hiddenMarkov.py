from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np
from numba import njit, float64, int32, boolean

# ----------------------------------------------------------------------
# Constants & Kernels (Converted to Numpy Arrays for Numba)
# Format: [dx, dy, probability]
# ----------------------------------------------------------------------

# We convert the dictionaries to Nx3 arrays for fast iteration in Numba
HEAR_KERNEL_DATA = np.array([
    [-2, -2, 0.00], [-1, -2, 0.10], [0, -2, 0.10], [1, -2, 0.10], [2, -2, 0.00],
    [-2, -1, 0.10], [-1, -1, 0.25], [0, -1, 0.50], [1, -1, 0.25], [2, -1, 0.10],
    [-2,  0, 0.10], [-1,  0, 0.50], [0,  0, 0.00], [1,  0, 0.50], [2,  0, 0.10],
    [-2,  1, 0.10], [-1,  1, 0.25], [0,  1, 0.50], [1,  1, 0.25], [2,  1, 0.10],
    [-2,  2, 0.00], [-1,  2, 0.10], [0,  2, 0.10], [1,  2, 0.10], [2,  2, 0.00],
], dtype=np.float64)

FEEL_KERNEL_DATA = np.array([
    [-1, -1, 0.15], [0, -1, 0.30], [1, -1, 0.15],
    [-1,  0, 0.30], [0,  0, 0.00], [1,  0, 0.30],
    [-1,  1, 0.15], [0,  1, 0.30], [1,  1, 0.15],
], dtype=np.float64)

# ----------------------------------------------------------------------
# Numba Logic
# ----------------------------------------------------------------------

@njit(fastmath=True, cache=True)
def fast_update_grid(
    p_grid: np.ndarray, 
    mask_grid: np.ndarray,
    px: int, 
    py: int, 
    hear_kernel: np.ndarray, 
    feel_kernel: np.ndarray, 
    heard: bool, 
    felt: bool
):
    """
    Updates a probability grid in-place based on sensor data.
    This replaces _get_likelihood_grid and the multiplication steps.
    """
    dim = p_grid.shape[0]

    # --- 1. Apply Hearing ---
    if heard:
        # Sparsity Optimization: 
        # If we heard something, the trap MUST be in the kernel range.
        # Everything else becomes 0. We create a temp buffer for the valid area.
        new_grid = np.zeros_like(p_grid)
        
        for i in range(hear_kernel.shape[0]):
            dx = int(hear_kernel[i, 0])
            dy = int(hear_kernel[i, 1])
            p_val = hear_kernel[i, 2]
            
            tx, ty = px - dx, py - dy
            
            if 0 <= tx < dim and 0 <= ty < dim:
                # P(new) = P(old) * P(sensor|loc)
                new_grid[tx, ty] = p_grid[tx, ty] * p_val
        
        # Swap back to p_grid
        p_grid[:] = new_grid[:]
        
    else:
        # If we did NOT hear, we multiply existing probabilities by (1 - p).
        # Optimization: We only need to touch the cells inside the kernel.
        # Cells outside the kernel are multiplied by 1.0 (unchanged).
        for i in range(hear_kernel.shape[0]):
            dx = int(hear_kernel[i, 0])
            dy = int(hear_kernel[i, 1])
            p_val = hear_kernel[i, 2]
            
            tx, ty = px - dx, py - dy
            
            if 0 <= tx < dim and 0 <= ty < dim:
                p_grid[tx, ty] *= (1.0 - p_val)

    # --- 2. Apply Feeling ---
    if felt:
        new_grid = np.zeros_like(p_grid)
        for i in range(feel_kernel.shape[0]):
            dx = int(feel_kernel[i, 0])
            dy = int(feel_kernel[i, 1])
            p_val = feel_kernel[i, 2]
            
            tx, ty = px - dx, py - dy
            if 0 <= tx < dim and 0 <= ty < dim:
                new_grid[tx, ty] = p_grid[tx, ty] * p_val
        p_grid[:] = new_grid[:]
    else:
        for i in range(feel_kernel.shape[0]):
            dx = int(feel_kernel[i, 0])
            dy = int(feel_kernel[i, 1])
            p_val = feel_kernel[i, 2]
            
            tx, ty = px - dx, py - dy
            if 0 <= tx < dim and 0 <= ty < dim:
                p_grid[tx, ty] *= (1.0 - p_val)

    # --- 3. Normalize ---
    total = np.sum(p_grid)
    if total > 0:
        p_grid /= total
    else:
        # Degenerate case: reset to uniform based on mask
        # Convert boolean mask to float in-place logic
        count = 0.0
        for r in range(dim):
            for c in range(dim):
                if mask_grid[r, c]:
                    p_grid[r, c] = 1.0
                    count += 1.0
                else:
                    p_grid[r, c] = 0.0
        if count > 0:
            p_grid /= count

@njit(fastmath=True, cache=True)
def fast_mark_safe(p_grid, x, y):
    if 0 <= x < p_grid.shape[0] and 0 <= y < p_grid.shape[1]:
        p_grid[x, y] = 0.0
        total = np.sum(p_grid)
        if total > 0:
            p_grid /= total

# ----------------------------------------------------------------------
# Main Class
# ----------------------------------------------------------------------

class TrapdoorBelief:
    """
    Exact Bayesian belief over the two trapdoors using Numba for speed.
    """

    def __init__(self, map_size: int):
        self.map_size = map_size
        
        # Pre-compute masks 
        self.mask_even = np.zeros((map_size, map_size), dtype=bool)
        self.mask_odd = np.zeros((map_size, map_size), dtype=bool)
        
        for x in range(map_size):
            for y in range(map_size):
                if (x + y) % 2 == 0:
                    self.mask_even[x, y] = True
                else:
                    self.mask_odd[x, y] = True

        # Initialize beliefs
        self.p_even = np.zeros((map_size, map_size), dtype=np.float64)
        self.p_odd  = np.zeros((map_size, map_size), dtype=np.float64)
        
        self._init_prob()

    def _init_prob(self) -> None:
        """Reset beliefs to match the trapdoor sampling distribution."""
        dim = self.map_size
        weights = np.zeros((dim, dim), dtype=np.float64)
        
        # Ring 2 -> w=1
        if dim >= 6:
            weights[2:dim-2, 2:dim-2] = 1.0
        
        # Inner core -> w=2
        if dim >= 8:
            weights[3:dim-3, 3:dim-3] = 2.0

        # Apply to Even
        self.p_even = weights * self.mask_even
        sum_even = np.sum(self.p_even)
        if sum_even > 0: self.p_even /= sum_even

        # Apply to Odd
        self.p_odd = weights * self.mask_odd
        sum_odd = np.sum(self.p_odd)
        if sum_odd > 0: self.p_odd /= sum_odd

    def reset(self) -> None:
        self._init_prob()

    def update(
        self,
        player_pos: Tuple[int, int],
        sensor_data: List[Tuple[bool, bool]],
    ) -> None:
        """
        Update beliefs given current player position and latest senses.
        sensor_data[0] = (heard_even, felt_even)
        sensor_data[1] = (heard_odd,  felt_odd)
        """
        (heard_e, felt_e), (heard_o, felt_o) = sensor_data
        px, py = player_pos

        # Update Even Grid
        fast_update_grid(
            self.p_even, self.mask_even,
            px, py, 
            HEAR_KERNEL_DATA, FEEL_KERNEL_DATA, 
            heard_e, felt_e
        )

        # Update Odd Grid
        fast_update_grid(
            self.p_odd, self.mask_odd,
            px, py, 
            HEAR_KERNEL_DATA, FEEL_KERNEL_DATA, 
            heard_o, felt_o
        )

    def mark_safe(self, pos: Tuple[int, int]) -> None:
        x, y = pos
        if (x + y) % 2 == 0:
            fast_mark_safe(self.p_even, x, y)
        else:
            fast_mark_safe(self.p_odd, x, y)

    def mark_safes(self, positions: List[Tuple[int, int]]) -> None:
        for pos in positions:
            self.mark_safe(pos)

    def set_trapdoor(self, pos: Tuple[int, int]) -> None:
        x, y = pos
        if (x + y) % 2 == 0:
            self.p_even.fill(0.0)
            self.p_even[x, y] = 1.0
        else:
            self.p_odd.fill(0.0)
            self.p_odd[x, y] = 1.0

    def prob_at(self, pos: Tuple[int, int]) -> float:
        x, y = pos
        if not (0 <= x < self.map_size and 0 <= y < self.map_size):
            return 0.0
        if (x + y) % 2 == 0:
            return float(self.p_even[x, y])
        else:
            return float(self.p_odd[x, y])

    def most_likely_even(self) -> Tuple[Tuple[int, int], float] | None:
        flat_idx = np.argmax(self.p_even)
        idx = np.unravel_index(flat_idx, self.p_even.shape)
        p = self.p_even[idx]
        if p == 0: return None
        return (int(idx[0]), int(idx[1])), float(p)

    def most_likely_odd(self) -> Tuple[Tuple[int, int], float] | None:
        flat_idx = np.argmax(self.p_odd)
        idx = np.unravel_index(flat_idx, self.p_odd.shape)
        p = self.p_odd[idx]
        if p == 0: return None
        return (int(idx[0]), int(idx[1])), float(p)

    def get_maps(self) -> Tuple[Dict[Tuple[int, int], float], Dict[Tuple[int, int], float]]:
        d_even = {}
        d_odd = {}
        rows, cols = np.nonzero(self.p_even)
        for r, c in zip(rows, cols):
            d_even[(r, c)] = float(self.p_even[r, c])
        rows, cols = np.nonzero(self.p_odd)
        for r, c in zip(rows, cols):
            d_odd[(r, c)] = float(self.p_odd[r, c])
        return d_even, d_odd

    def debug_print(self, precision: int = 3) -> None:
        print("\n=== Even Belief ===")
        print(np.round(self.p_even, precision).T) 
        print("\n=== Odd Belief ===")
        print(np.round(self.p_odd, precision).T)