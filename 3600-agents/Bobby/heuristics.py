import math
from game import board as board_mod  # type: ignore
from .voronoi import VoronoiInfo, OWNER_ME  # adjust import path as needed

def evaluate(self, cur_board: board_mod.Board, vor: VoronoiInfo) -> float:
        """
        Bobby's scalar evaluation.
        Always from *my* POV (current chicken_player), regardless of depth.

        Components:
          - space_term:   parity+corner-weighted Voronoi score + small frontier bonus
          - mat_term:     eggs_laid difference (already includes corner bonus & traps)
          - risk_term:    penalty based on *my* trapdoor entropy in my Voronoi region
        """

        # Material: eggs laid already accounts for corners + trap teleports
        my_eggs  = cur_board.chicken_player.eggs_laid
        opp_eggs = cur_board.chicken_enemy.eggs_laid
        mat_diff = my_eggs - opp_eggs

        # Space control (parity + corner weighted)
        space_score = vor.vor_score  # my_voronoi - opp_voronoi

        # Frontier bonus: having some contested frontier early is good
        moves_left = cur_board.MAX_TURNS - cur_board.turn_count
        total_turns = cur_board.MAX_TURNS
        phase = max(0.0, min(1.0, moves_left / total_turns))  # 1.0 = opening, 0.0 = final turns

        frontier_bonus = 0.5 * phase * vor.contested

        # Weights (tunable)
        # Early: space matters more, late: material dominates
        W_SPACE_EARLY = 3.0
        W_SPACE_LATE  = 0.5
        W_MAT_EARLY   = 8.0
        W_MAT_LATE    = 40.0

        w_space = W_SPACE_LATE + phase * (W_SPACE_EARLY - W_SPACE_LATE)
        w_mat   = W_MAT_LATE   + phase * (W_MAT_EARLY   - W_MAT_LATE)

        # Risk term from my trap belief
        risk_term = self._my_entropy_penalty(cur_board, vor, phase)

        space_term = w_space * space_score + frontier_bonus
        mat_term   = w_mat   * mat_diff

        return space_term + mat_term + risk_term


def _my_entropy_penalty(
        self,
        cur_board: board_mod.Board,
        vor: VoronoiInfo,
        phase: float,
    ) -> float:
        """
        Penalize positions where:
          - a lot of trap probability mass lies inside my Voronoi region, and
          - that mass is high-entropy (spread out, i.e., hard to avoid).

        We restrict to squares owned by OWNER_ME in vor.owner.
        """

        dim = cur_board.game_map.MAP_SIZE

        # 1) Collect probabilities in my Voronoi region
        probs = []
        total_mass = 0.0

        for x in range(dim):
            for y in range(dim):
                if vor.owner[x][y] != OWNER_ME:
                    continue
                p = self.trap_belief.prob_at((x, y))
                if p <= 0.0:
                    continue
                probs.append(p)
                total_mass += p

        if total_mass <= 0.0 or not probs:
            return 0.0

        # 2) Normalize to get a distribution over my region
        norm_probs = [p / total_mass for p in probs]

        # 3) Shannon entropy H = -sum p_i log p_i
        H = 0.0
        for q in norm_probs:
            if q <= 0.0:
                continue
            H -= q * math.log(q)  # natural log; scale absorbed into weight

        # 4) Combine mass + entropy into a single penalty
        # More mass → worse.
        # More entropy (spread) → also worse (harder to dodge).
        # Early in the game, we care more about risk than in the final turns.
        MASS_COEF    = 40.0     # how bad it is to have trap mass in my region
        ENTROPY_COEF = 12.0     # how bad it is for that mass to be spread out

        # Weight risk more in earlier phase; still nonzero late.
        phase_risk = 0.3 + 0.7 * phase

        penalty = phase_risk * (MASS_COEF * total_mass + ENTROPY_COEF * H)

        # Evaluation is "good is high", so risk is subtracted.
        return -penalty