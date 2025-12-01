from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import numpy as np

# ----------------------------------------------------------------------
# Sensor kernels
# ----------------------------------------------------------------------

# Coordinates are (dx, dy) = (player_x - trap_x, player_y - trap_y).
# We interpret this as: if player is at P, a trap at T = P - (dx, dy) contributes probability p.
PROB_HEAR_KERNEL: Dict[Tuple[int, int], float] = {
    (-2, -2): 0.00, (-1, -2): 0.10, (0, -2): 0.10, (1, -2): 0.10, (2, -2): 0.00,
    (-2, -1): 0.10, (-1, -1): 0.25, (0, -1): 0.50, (1, -1): 0.25, (2, -1): 0.10,
    (-2,  0): 0.10, (-1,  0): 0.50, (0,  0): 0.00, (1,  0): 0.50, (2,  0): 0.10,
    (-2,  1): 0.10, (-1,  1): 0.25, (0,  1): 0.50, (1,  1): 0.25, (2,  1): 0.10,
    (-2,  2): 0.00, (-1,  2): 0.10, (0,  2): 0.10, (1,  2): 0.10, (2,  2): 0.00,
}

PROB_FEEL_KERNEL: Dict[Tuple[int, int], float] = {
    (-1, -1): 0.15, (0, -1): 0.30, (1, -1): 0.15,
    (-1,  0): 0.30, (0,  0): 0.00, (1,  0): 0.30,
    (-1,  1): 0.15, (0,  1): 0.30, (1,  1): 0.15,
}

class TrapdoorBelief:
    """
    Exact Bayesian belief over the two trapdoors using NumPy for vectorization.

    Hidden state:
        T_e ∈ {even squares}
        T_o ∈ {odd  squares}

    We maintain two NxN grids:
        self.p_even[x, y]
        self.p_odd[x, y]
    """

    def __init__(self, map_size: int):
        self.map_size = map_size
        
        # Pre-compute masks for valid even/odd squares
        self.mask_even = np.zeros((map_size, map_size), dtype=bool)
        self.mask_odd = np.zeros((map_size, map_size), dtype=bool)
        
        for x in range(map_size):
            for y in range(map_size):
                if (x + y) % 2 == 0:
                    self.mask_even[x, y] = True
                else:
                    self.mask_odd[x, y] = True

        # Initialize beliefs
        self.p_even = np.zeros((map_size, map_size), dtype=float)
        self.p_odd  = np.zeros((map_size, map_size), dtype=float)
        
        self._init_prob()

    def _init_prob(self) -> None:
        """Reset beliefs to match the trapdoor sampling distribution."""
        dim = self.map_size
        
        # Create a weights grid based on the rules
        # Edge (0) -> w=0
        # Inside edge (1) -> w=0
        # Ring 2 (2) -> w=1
        # Inner core (3+) -> w=2
        weights = np.zeros((dim, dim), dtype=float)
        
        # Fill weights based on "rings" from the edge
        # We can just iterate or slice. Slicing is cleaner.
        # Initialize ring 2 (indices 2 to dim-3 inclusive) with 1.0
        if dim >= 6:
            weights[2:dim-2, 2:dim-2] = 1.0
        
        # Initialize inner core (indices 3 to dim-4 inclusive) with 2.0
        if dim >= 8:
            weights[3:dim-3, 3:dim-3] = 2.0

        # Apply to Even
        self.p_even = weights * self.mask_even
        sum_even = np.sum(self.p_even)
        if sum_even > 0:
            self.p_even /= sum_even

        # Apply to Odd
        self.p_odd = weights * self.mask_odd
        sum_odd = np.sum(self.p_odd)
        if sum_odd > 0:
            self.p_odd /= sum_odd

    def reset(self) -> None:
        self._init_prob()

    # ------------------------------------------------------------------
    # Core update (Bayes filter)
    # ------------------------------------------------------------------

    def _get_likelihood_grid(
        self, 
        player_pos: Tuple[int, int], 
        kernel: Dict[Tuple[int, int], float], 
        sensed: bool
    ) -> np.ndarray:
        """
        Constructs a likelihood grid for the entire board based on one sensor reading.
        
        If sensed=True (Heard/Felt):
           L(trap_pos) = P(sensed | trap_pos) = kernel_value
        If sensed=False (Not Heard/Not Felt):
           L(trap_pos) = 1 - P(sensed | trap_pos) = 1 - kernel_value
        """
        dim = self.map_size
        px, py = player_pos
        
        # Start with default likelihood
        # If we heard something, default likelihood for squares outside kernel is 0.
        # If we didn't hear, default likelihood for squares outside kernel is 1.
        if sensed:
            L = np.zeros((dim, dim), dtype=float)
        else:
            L = np.ones((dim, dim), dtype=float)

        # Apply kernel
        # dx = px - tx  => tx = px - dx
        # dy = py - ty  => ty = py - dy
        for (dx, dy), p_val in kernel.items():
            tx, ty = px - dx, py - dy
            
            # Check bounds
            if 0 <= tx < dim and 0 <= ty < dim:
                if sensed:
                    L[tx, ty] = p_val
                else:
                    L[tx, ty] = 1.0 - p_val
                    
        return L

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

        # --- EVEN UPDATE ---
        L_hear_e = self._get_likelihood_grid(player_pos, PROB_HEAR_KERNEL, heard_e)
        L_feel_e = self._get_likelihood_grid(player_pos, PROB_FEEL_KERNEL, felt_e)
        
        # Posterior = Prior * Likelihood * Likelihood
        self.p_even *= L_hear_e
        self.p_even *= L_feel_e
        
        # Normalize
        total_e = np.sum(self.p_even)
        if total_e > 0:
            self.p_even /= total_e
        else:
            # Degenerate case (shouldn't happen with valid logic), reset to uniform over valid mask
            self.p_even = self.mask_even.astype(float)
            self.p_even /= np.sum(self.p_even)

        # --- ODD UPDATE ---
        L_hear_o = self._get_likelihood_grid(player_pos, PROB_HEAR_KERNEL, heard_o)
        L_feel_o = self._get_likelihood_grid(player_pos, PROB_FEEL_KERNEL, felt_o)
        
        self.p_odd *= L_hear_o
        self.p_odd *= L_feel_o
        
        # Normalize
        total_o = np.sum(self.p_odd)
        if total_o > 0:
            self.p_odd /= total_o
        else:
            self.p_odd = self.mask_odd.astype(float)
            self.p_odd /= np.sum(self.p_odd)

    def mark_safe(self, pos: Tuple[int, int]) -> None:
        """
        Set probability of `pos` to 0 (we stepped there and didn't die).
        Renormalize.
        """
        x, y = pos
        if not (0 <= x < self.map_size and 0 <= y < self.map_size):
            return

        if (x + y) % 2 == 0:
            self.p_even[x, y] = 0.0
            s = np.sum(self.p_even)
            if s > 0: self.p_even /= s
        else:
            self.p_odd[x, y] = 0.0
            s = np.sum(self.p_odd)
            if s > 0: self.p_odd /= s

    def mark_safes(self, positions: List[Tuple[int, int]]) -> None:
        for pos in positions:
            self.mark_safe(pos)

    def set_trapdoor(self, pos: Tuple[int, int]) -> None:
        """
        Collapse belief: we know the trap is exactly at `pos`.
        """
        x, y = pos
        if (x + y) % 2 == 0:
            self.p_even.fill(0.0)
            self.p_even[x, y] = 1.0
        else:
            self.p_odd.fill(0.0)
            self.p_odd[x, y] = 1.0

    # ------------------------------------------------------------------
    # Query interface
    # ------------------------------------------------------------------

    def prob_at(self, pos: Tuple[int, int]) -> float:
        x, y = pos
        if not (0 <= x < self.map_size and 0 <= y < self.map_size):
            return 0.0
        if (x + y) % 2 == 0:
            return float(self.p_even[x, y])
        else:
            return float(self.p_odd[x, y])

    def most_likely_even(self) -> Tuple[Tuple[int, int], float] | None:
        # np.argmax returns flat index, unravel to get (x,y)
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
        """
        Compatibility layer: converts numpy arrays back to dicts if other code needs it.
        Ideally, you update other code to use numpy arrays directly.
        """
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
        # Transpose (.T) so that x is column and y is row (visual match to board)
        print(np.round(self.p_even, precision).T) 
        print("\n=== Odd Belief ===")
        print(np.round(self.p_odd, precision).T)
