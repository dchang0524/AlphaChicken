# Heuristic Overview

AlphaChicken’s evaluation function approximates the expected final egg difference by combining spatial control, reachability analysis, trapdoor risk modeling, and dynamic weighting tied to game phase. It integrates these terms directly inside alpha–beta search so that path-dependent factors (like trapdoor risk) are accounted for consistently across the search tree.

## 1. Material & Reachability Terms
### 1.1 Base Egg Difference

Reward current egg lead:

my_eggs – opp_eggs

Scaled lightly to avoid early-game distortion.

### 1.2 Potential Egg Gain

Estimate the number of eggs each side can still collect:

- Eggs in my Voronoi region (I’m strictly closer).

- Eggs in contested regions, weighted by distance and likelihood of winning the race.

- Eggs in my unreachable-by-opponent region, weighted higher.

- Capped by available moves (min(moves_left/2, egg_count)).

This approximates the upper bound on future material.

Potential Improvements: Build a probabilisitc model based on the difference of the distance to that square from both players to define ownership. Create a better approximation for eggs you can collect given the number of moves, possibly by computing each connected component of unclaimed eggs and then calculating the distances between each component.

### 1.3 Distance-to-Egg Bonus (Endgame)

When the nearest egg is far and alpha–beta cannot see the capture sequence:

- Add a small bonus inversely proportional to distance.

- Allows search to choose lines that actually reach eggs later.

## 2. Spatial Control & Region Structure
### 2.1 Voronoi Space Difference

The core spatial metric:

Compute per-player Voronoi regions (distance-only BFS). 

A square where I can reach first before my opponent belongs to my Voronoi region. 

Technically, if there were no trapdoors or turn limit, you can't overcome this space difference. 

Reward:

- Larger owned region
- Better frontier control
- Strategic positions (e.g., central reachability)
- Voronoi naturally encodes mobility, territory, and long-term potential eggs.

### 2.2 Contested Squares

Squares where both players reach with similar distance.
These matter because they determine:

Where infiltration is possible

Where turds and tempo actually change the game

AlphaChicken evaluates:

- Count of contested squares
- My distance to each
- Opponent’s distance(which is equal or differs by 1, by definition)
→ Pushes toward controlling relevant “front lines.”

### 2.3 Contested Region Significance

Not all contested regions are equally valuable.

Define region significance:

**significance** = (# eggs in this region) / (total remaining eggs)

Apply this to 2.2.


This prevents the bot from over-fighting over meaningless regions, one of the biggest weaknesses of naïve Voronoi-based bots.

### 2.4 Fragmentation & Openness

Board structure metrics:

**Openness** = (# contested squares) / 8

**Fragmentation** = Manhattan distance between extreme frontier squares

Used only to scale dynamic weights (see §5).
Fragmented boards → fight only in key regions
Closed boards → prioritize guaranteed egg collection

## 3. Trapdoor Risk Modeling
### 3.1 Path-Dependent Trapdoor Risk

Using a Bayesian trapdoor belief model:

Each path in alpha–beta accumulates probability of stepping on trapdoors

Weighted penalty added to leaf evaluations

Accounts for:

- Sensor readings
- Equidistant trapdoor ambiguity
- Movement through high-risk lines

This prevents superficially “good” lines that are suicidal in reality.

If steping on a known trapdoor, we can directly simulate its effects.

### 3.2 Global Trap Hazard Term

A small constant factor:

trap_weight × (4 + 8 × openness × voronoi_factor × dist_from_start / 2)

This is a cheap approximation of how much space and eggs you will lose from stepping on a trapdoor.

### 3.3 Entropy

We didn't get to implement this, but we want to give some sort of bonus for visiting squares that would clarify our trapdoor distributions(reducing entropy). This could be done by approximating entropy gained from visiting a square and making this bonus term path-dependent. To prevent rushing the center(which we might end up just stepping on a trapdoor without getting enough information), we could also keep a visited count array to give some sort of confidence(However, we wouldn't need this if we implemented a way to deal with high variance moves in expectimax).
 

Represents the weight of penalty for 3.1.

## 4. Turd Management
### 4.1 Positive Turd Savings

Don’t waste turds for no strategic gain:

Reward keeping unused turds (weight × turds_left)

### 4.2 Wasted Turd Penalty

A turd is penalized if it contributes nothing:

- Surrounded entirely by unreachable squares

- Blocks space already fully owned

- Trapped inside self-contained walls

- Or reinforces a region with redundant coverage

Cheap approximate tests avoid expensive recomputation.

### 4.3 Blocks (Successful Turd Usage)

Reward “hard territory” created by turds:

- Squares opponent can no longer reach

- Reductions in opponent Voronoi region

- Creation of cutoff regions

Blocking is essentially “permanent Voronoi.”

This is also done by giving higher weight to blocked eggs in Potential Egg Gain. 

## 5. Dynamic Weighting

Game phase changes which heuristics matter.

### Early Game

- Space > Material
- Expansion and frontier contests matter most
- Avoid early corner-egg greed (common MAX TA bot weakness)

### Midgame

- Contested region significance dominates
- Trapdoor risk sharply increases impact

### Late Game
- Material > Space
- Distance-to-egg bonus active
- Endgame egg races decided mostly by distances & remaining moves

### Transitions

Weight transitions depend on:
- Turn number
- Openness
- Fragmentation
- Remaining eggs

A linear + structural blend is used for stable phase-change behavior.

## 6. Loss Prevention Module

Explicit high-penalty triggers for forced losses:

### 6.1 Region Closed & Losing on Material

If both sides are sealed off and my total reachable eggs < opponent, apply heavy negative.

### 6.2 No-Move → +5/-5 Rule Check

Terminal states under “no moves left” rule penalized correctly.

### 6.3 Turn 80 Finalization

If final egg count is losing once turns are exhausted.

### 6.4 Theoretical Maximum Loss Detection

If even under optimal collection (corners + race-in + best-case path) I still lose:

Apply strong negative signal

Forces alpha–beta to avoid dead lines earlier

## 7. Final Evaluation Formula (Conceptual)

The evaluation approximates Expected Final Egg Difference:

Score =
    BaseEggDiff
  + PotentialEggs
  - TrapdoorRisk
  - WastedTurdPenalty
  + TurdSavings
  + EndgameEggDistanceBonus
  + WeightedContestedSquareTerm


All modulated through dynamic weights depending on game phase, openness, and fragmentation.

## 8. Current Known Weaknesses

Overfighting trivial contested squares
→ Caused by insufficient region significance discounting.

Slow conversion in winning positions
→ Sometimes prefers space/control over guaranteed eggs.

Occasional turd inefficiency
→ Needs better redundancy detection.

Trapdoor-induced volatility
→ Voronoi boundaries jump under small positional changes. This makes techniques like futility and aspiration windows impractical to implement.

# Algorithmic Overview

AlphaChicken combines classic game-tree search with probabilistic modeling and low-level performance tricks to handle the noisy, trapdoor-heavy dynamics of ChickenFight.

## Search Framework: Negamax / Minimax

The core search is a depth-limited **negamax** (minimax) with alpha–beta pruning:

- Uniform scoring: positions are always evaluated from the current player’s perspective and sign-flipped on recursion.
- Alpha–beta pruning: branches that cannot improve the current best value are cut early.
- Loss-prevention hooks: terminal or provably-losing positions are detected and given large negative scores to stabilize decision-making.

This search runs continuously under a strict time budget, so almost all other components are designed around making each node as cheap as possible.

## Iterative Deepening & Time Management

Search is driven by **iterative deepening**:

- Start at shallow depth (e.g., 1–2 plies), then repeatedly increase depth as long as time allows.
- The best move from the previous iteration is used as the first candidate in the next iteration for strong move ordering.
- Time checks are baked into the search loop; when time is nearly exhausted, the engine immediately returns the best result from the last completed depth.

This gives:
- Anytime behavior (always have a valid move),
- Better move ordering over time (feeding TT + history),
- Graceful degradation instead of random timeouts.

## Transposition Table (TT) & Zobrist Hashing

AlphaChicken uses a **transposition table** to cache search results for repeated positions:

- Each node stores:
  - Zobrist hash
  - Best move
  - Search depth
  - Node type (exact / lower bound / upper bound)
  - Score in a risk-aware domain
  - Metadata for when it was computed (for re-use across iterations)

### Zobrist Hashing (Board vs Voronoi)

Two independent Zobrist systems are maintained:

1. **Board Hash**
   - Encodes positions of both players, eggs, turds, known trapdoors, side-to-move, and turn.
   - Used as the TT key; updated incrementally when applying moves.

2. **Voronoi / Eval Hash**
   - Encodes derived features related to Voronoi regions, frontiers, and other expensive-to-recompute spatial features.
   - Allows caching and reusing evaluation-related structures without fully recomputing BFS for every node.

This separation avoids recomputing spatial structures unnecessarily while keeping TT keys focused on true game state.


## Trapdoor Belief Model (HMM-style)

Trapdoors are modeled with a **belief distribution over squares**, updated in a Bayesian fashion. The belief state behaves like a small Hidden Markov Model over “trapdoor presence” at each tile, with these inputs:

- **Positive signals**  
  E.g., hearing/feeling near a trapdoor. These update nearby tiles according to a fixed sensor kernel.

- **Negative information (absence of signals)**  
  If we (or the opponent) stand in a region where a trapdoor *would* have produced a signal but none occurred, we down-weight those nearby tiles. Absence of signal is still information.

- **Our movement**  
  - When we step on a square and don’t fall, that square’s probability drops (often to zero, depending on rules).
  - Our path through the board carves out “safe corridors” where trapdoors almost surely are not.
  - Repeated visits without any signals further reduce probabilities in those neighborhoods.

- **Opponent movement**  
  - If the opponent steps on a square and doesn’t fall, that square’s probability also drops.
  - Their chosen routes give indirect info: they tend not to walk through regions they believe are lethal, so their avoidance pattern is weak evidence about trapdoor placement.

The search uses this belief in two ways:

1. **Path-dependent risk in search**  
   Along each search line, we accumulate the probability of stepping on trapdoors based on the belief + the *exact* move sequence for both players. That accumulated risk is penalized directly in the search value, not just at static leaf evaluation.

2. **Evaluation-time risk term**  
   At leaf nodes, a global risk term approximates long-run trap exposure based on:
   - Board openness
   - Typical distances to unexplored areas
   - The current belief distribution

Net effect: the bot “knows” when a line is probabilistically suicidal even if it looks spatially good, and it learns from both our own pathing and the opponent’s pathing.


## Expectimax for Uncertainty

Not all branches are purely adversarial:

- Trapdoor outcomes and sensor signals introduce **stochasticity**.
- Opponent behavior may be approximated as a mixture of adversarial and “typical” moves.

At the start of each search, we collapse high-probability trapdoor squares into fixed trap assumptions and generate a small set of plausible trap configurations. We run a full minimax search on each configuration, obtain a value for every action, and then select the move with the highest posterior‐weighted expected value. Any trapdoors outside this configuration set are still accounted for through a path-dependent risk term, ensuring the search incorporates the residual uncertainty that wasn’t explicitly enumerated.

## Bitboards & Array-Based BFS

The board is represented internally using **bitboards**:

- Squares are packed into fixed-size integer masks.
- Basic operations like:
  - adjacency,
  - reachability,
  - region counting,
  are done via bit operations and precomputed masks.

For **Voronoi and reachability**, the engine uses **array-based BFS** instead of Python sets/dicts:

- Preallocated arrays store:
  - distances,
  - queues,
  - ownership markers.
- BFS uses simple integer indices, no heap allocation in the inner loop.
- This is significantly faster and more cache-friendly than set-based BFS, which matters because Voronoi and distance fields are recomputed constantly.

Bitboards + array BFS make spatial computations fast enough to fit deep search under tight time constraints.

## Implementation & Optimization Stack

The implementation is split between **C++** and optimized **Python**:

- Performance-critical search routines (negamax + alpha–beta + TT) and some low-level board operations run in **C++**.
- Higher-level logic and experimentation (heuristics, evaluation tweaks, belief updates) live in **Python**.
- Hot Python kernels (like BFS variants or mass belief updates) are JIT-compiled using **Numba**, reducing overhead while keeping iteration speed for development.

This setup gives:
- C++-level performance on the critical path,
- Python-level flexibility for iterating on heuristics and modeling.
- 

## Known Issues

The biggest problem with the bot is that it won't avoid trapdoors even if it thinks there is a high probability that there is trapdoor in that square because we are simply taking the move that maximizes the expected value of the score. Since we're taking the best expected value,
if we have a move that has a 0.5 probability of +20 and 0.5 probability of -10, and a move that is +9 with probability 1 it will pick the risky move, which is not what we want. Some potential fixes we thought of that we haven't implemented yet are: 
- Instead of maximizing E[score], maximize E[f(score)], where f is concave and increasing. A good function would probably be f(x) = sign(x) * sqrt(|x|).
- We could give some sort of penalty for moves with high variance in scores.
- We could give extra weight to lines with negative scores.
- We could cap off the maximum score of winning lines.

# Agent History
1. Morphy
2. Tal
3. Bobby
4. Magnus
5. Hikaru
6. CPPHikaru
