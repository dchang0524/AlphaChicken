from __future__ import annotations
from typing import Dict, List, Tuple

#TODO: Change to numpy for speed up


# ----------------------------------------------------------------------
# Sensor kernels (from the spec figure)
# ----------------------------------------------------------------------

# prob_hear: 5x5 kernel around the trapdoor.
# Coordinates are (dx, dy) = (player_x - trap_x, player_y - trap_y).
PROB_HEAR_KERNEL: Dict[Tuple[int, int], float] = {
    (-2, -2): 0.00, (-1, -2): 0.10, (0, -2): 0.10, (1, -2): 0.10, (2, -2): 0.00,
    (-2, -1): 0.10, (-1, -1): 0.25, (0, -1): 0.50, (1, -1): 0.25, (2, -1): 0.10,
    (-2,  0): 0.10, (-1,  0): 0.50, (0,  0): 0, (1,  0): 0.50, (2,  0): 0.10,
    (-2,  1): 0.10, (-1,  1): 0.25, (0,  1): 0.50, (1,  1): 0.25, (2,  1): 0.10,
    (-2,  2): 0.00, (-1,  2): 0.10, (0,  2): 0.10, (1,  2): 0.10, (2,  2): 0.00,
}

# prob_feel: 3x3 kernel around the trapdoor.
PROB_FEEL_KERNEL: Dict[Tuple[int, int], float] = {
    (-1, -1): 0.15, (0, -1): 0.30, (1, -1): 0.15,
    (-1,  0): 0.30, (0,  0): 0, (1,  0): 0.30,
    (-1,  1): 0.15, (0,  1): 0.30, (1,  1): 0.15,
}


class TrapdoorBelief:
    """
    Exact Bayesian belief over the two trapdoors.

    Hidden state:
        T_e ∈ {even squares}
        T_o ∈ {odd  squares}

    We maintain:
        p_even[s] = P(T_e = s | history)
        p_odd[s]  = P(T_o = s | history)

    Each turn we observe:
        sensor_data[0] = (heard_even, felt_even)
        sensor_data[1] = (heard_odd,  felt_odd)

    and update each parity independently using Bayes.
    Complexity per update: O(#board_squares).
    """

    def __init__(self, map_size: int):
        self.map_size = map_size

        # Partition board into even/odd squares by parity of x+y
        self.even_squares: List[Tuple[int, int]] = []
        self.odd_squares: List[Tuple[int, int]] = []

        for x in range(map_size):
            for y in range(map_size):
                if (x + y) % 2 == 0:
                    self.even_squares.append((x, y))
                else:
                    self.odd_squares.append((x, y))

        self.collapsed_known_traps: set[Tuple[int, int]] = set()

        self._init_prob()

    # ------------------------------------------------------------------
    # Initialization / reset
    # ------------------------------------------------------------------

    def _init_prob(self) -> None:
        """Reset beliefs to match the trapdoor sampling distribution."""

        dim = self.map_size

        weights_even: Dict[Tuple[int, int], float] = {}
        weights_odd: Dict[Tuple[int, int], float] = {}

        for x in range(dim):
            for y in range(dim):
                # Compute the same weight pattern as in TrapdoorManager.choose_trapdoors
                w = 0.0

                # Outer allowed region: [2 : dim-2] in both axes => weight 1
                if 2 <= x < dim - 2 and 2 <= y < dim - 2:
                    w = 1.0

                # Inner region: [3 : dim-3] => override with weight 2
                if 3 <= x < dim - 3 and 3 <= y < dim - 3:
                    w = 2.0

                if w == 0.0:
                    # These squares never get a trapdoor in the generator, so prior should be 0.
                    continue

                if (x + y) % 2 == 0:
                    weights_even[(x, y)] = w
                else:
                    weights_odd[(x, y)] = w

        # Normalize per parity
        sum_even = sum(weights_even.values())
        sum_odd = sum(weights_odd.values())

        if sum_even > 0:
            self.p_even = {pos: w / sum_even for pos, w in weights_even.items()}
        else:
            self.p_even = {}

        if sum_odd > 0:
            self.p_odd = {pos: w / sum_odd for pos, w in weights_odd.items()}
        else:
            self.p_odd = {}


    def reset(self) -> None:
        """Public reset, if you ever replay a game with the same agent."""
        self._init_uniform()

    # ------------------------------------------------------------------
    # Core update (Bayes filter)
    # ------------------------------------------------------------------

    @staticmethod
    def _likelihood_for_square(
        trap_pos: Tuple[int, int],
        player_pos: Tuple[int, int],
        heard: bool,
        felt: bool,
    ) -> float:
        """
        Compute P(heard, felt | trap at trap_pos, player at player_pos)
        using the prob_hear / prob_feel kernels and dx, dy offsets.
        """
        dx = player_pos[0] - trap_pos[0]
        dy = player_pos[1] - trap_pos[1]

        ph = PROB_HEAR_KERNEL.get((dx, dy), 0.0)
        pf = PROB_FEEL_KERNEL.get((dx, dy), 0.0)

        lh = ph if heard else (1.0 - ph)
        lf = pf if felt else (1.0 - pf)

        return lh * lf

    @staticmethod
    def _bayes_update_map(
        prior: Dict[Tuple[int, int], float],
        player_pos: Tuple[int, int],
        heard: bool,
        felt: bool,
    ) -> Dict[Tuple[int, int], float]:
        """
        One Bayes step for a single parity:
            posterior(s) ∝ prior(s) * P(obs | trap = s)
        """
        posterior: Dict[Tuple[int, int], float] = {}

        for pos, p_old in prior.items():
            L = TrapdoorBelief._likelihood_for_square(pos, player_pos, heard, felt)
            posterior[pos] = p_old * L

        Z = sum(posterior.values())
        if Z <= 0.0:
            # Degenerate case: fall back to uniform over the same support.
            n = len(posterior)
            if n == 0:
                return posterior
            u = 1.0 / n
            return {pos: u for pos in posterior}

        for pos in posterior:
            posterior[pos] /= Z

        return posterior

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

        # Even trap
        self.p_even = self._bayes_update_map(
            self.p_even, player_pos, heard_e, felt_e
        )

        # Odd trap
        self.p_odd = self._bayes_update_map(
            self.p_odd, player_pos, heard_o, felt_o
        )

    def _zero_and_renorm(self, pos: Tuple[int, int]) -> None:
        """
        Internal helper:
        - Set probability of `pos` to 0 in the appropriate parity map.
        - Renormalize that map so probabilities still sum to 1
          (condition on "trap is not here").
        """
        # Select parity map
        if (pos[0] + pos[1]) % 2 == 0:
            pmap = self.p_even
        else:
            pmap = self.p_odd

        if pos not in pmap:
            return  # already impossible or outside support

        old = pmap.pop(pos)
        remaining_mass = 1.0 - old

        if remaining_mass <= 0.0 or not pmap:
            # Degenerate (should not really happen unless we've
            # eliminated every possible square); in that case,
            # leave as-is or re-init prior if you prefer.
            return

        # Renormalize
        for k in pmap:
            pmap[k] /= remaining_mass

    def mark_safe(self, pos: Tuple[int, int]) -> None:
        """
        Incorporate the fact that `pos` has been stepped on by *someone*
        without triggering a trap. Hence, it CANNOT be the trapdoor.
        """
        self._zero_and_renorm(pos)

    def mark_safes(self, positions: List[Tuple[int, int]]) -> None:
        """
        Batch version for convenience.
        """
        for pos in positions:
            self.mark_safe(pos)


    # ------------------------------------------------------------------
    # Query interface
    # ------------------------------------------------------------------

    def prob_at(self, pos: Tuple[int, int]) -> float:
        """
        Return P("this square is the trapdoor of its parity").

        This is what you feed into your evaluation:
            - big penalty if prob_at(pos) is high
            - discount potential eggs behind high-prob squares
        """
        if (pos[0] + pos[1]) % 2 == 0:
            return self.p_even.get(pos, 0.0)
        else:
            return self.p_odd.get(pos, 0.0)

    def most_likely_even(self) -> Tuple[Tuple[int, int], float] | None:
        if not self.p_even:
            return None
        pos = max(self.p_even, key=self.p_even.get)
        return pos, self.p_even[pos]

    def most_likely_odd(self) -> Tuple[Tuple[int, int], float] | None:
        if not self.p_odd:
            return None
        pos = max(self.p_odd, key=self.p_odd.get)
        return pos, self.p_odd[pos]
    
    def set_trapdoor(self, pos: Tuple[int, int]) -> None:
        """
        Collapse the belief: we *know* there is a trapdoor at `pos`.

        If pos is even parity, that's the even trap.
        If pos is odd parity, that's the odd trap.
        """
        if (pos[0] + pos[1]) % 2 == 0:
            # Even trapdoor
            self.p_even = {pos: 1.0}
        else:
            # Odd trapdoor
            self.p_odd = {pos: 1.0}


    def get_maps(self) -> Tuple[Dict[Tuple[int, int], float], Dict[Tuple[int, int], float]]:
        """Optional helper: return full belief maps (for debugging / heatmaps)."""
        return self.p_even, self.p_odd
