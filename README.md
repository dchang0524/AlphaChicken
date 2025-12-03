# AlphaChicken Heuristic Overview

AlphaChicken’s evaluation function approximates the expected final egg difference by combining spatial control, reachability analysis, trapdoor risk modeling, and dynamic weighting tied to game phase. It integrates these terms directly inside alpha–beta search so that path-dependent factors (like trapdoor risk) are accounted for consistently across the search tree.

## 1. Material & Reachability Terms
### 1.1 Base Egg Difference

Reward current egg lead:

my_eggs – opp_eggs

Scaled lightly to avoid early-game distortion.

### 1.2 Potential Egg Gain

Estimate the number of eggs each side can still collect:

Eggs in my Voronoi region (I’m strictly closer).

Eggs in contested regions, weighted by distance and likelihood of winning the race.

Eggs in my unreachable-by-opponent region, weighted higher.

Capped by available moves (min(moves_left/2, egg_count)).

This approximates the upper bound on future material.

### 1.3 Distance-to-Egg Bonus (Endgame)

When the nearest egg is far and alpha–beta cannot see the capture sequence:

Add a small bonus inversely proportional to distance.

Allows search to choose lines that actually reach eggs later.

## 2. Spatial Control & Region Structure
### 2.1 Voronoi Space Difference

The core spatial metric:

Compute per-player Voronoi regions (distance-only BFS).

Reward:

Larger owned region

Better frontier control

Strategic positions (e.g., central reachability)

Voronoi naturally encodes mobility, territory, and long-term potential eggs.

### 2.2 Contested Squares

Squares where both players reach with similar distance.
These matter because they determine:

Where infiltration is possible

Where turds and tempo actually change the game

AlphaChicken evaluates:

Count of contested squares

My distance to each

Opponent’s distance
→ Pushes toward controlling relevant “front lines.”

### 2.3 Contested Region Significance

Not all contested regions are equally valuable.

Define region significance:

significance = (# eggs in this region) / (total remaining eggs)


Score contribution:

significance × distance × (1 - progression_factor)


This prevents the bot from over-fighting over meaningless regions, one of the biggest weaknesses of naïve Voronoi-based bots.

### 2.4 Fragmentation & Openness

Board structure metrics:

Openness = (# contested squares) / 8

Fragmentation = Manhattan distance between extreme frontier squares

Used only to scale dynamic weights (see §5).
Fragmented boards → fight only in key regions
Closed boards → prioritize guaranteed egg collection

## 3. Trapdoor Risk Modeling
### 3.1 Path-Dependent Trapdoor Risk

Using a Bayesian trapdoor belief model:

Each path in alpha–beta accumulates probability of stepping on trapdoors

Weighted penalty added to leaf evaluations

Accounts for:

Sensor readings

Equidistant trapdoor ambiguity

Movement through high-risk lines

This prevents superficially “good” lines that are suicidal in reality.

### 3.2 Global Trap Hazard Term

A small constant factor:

trap_weight × (4 + 8 × openness × voronoi_factor × dist_from_start / 2)


Represents long-run exposure to uncertain tiles.

## 4. Turd Management
### 4.1 Positive Turd Savings

Don’t waste turds for no strategic gain:

Reward keeping unused turds (weight × turds_left)

### 4.2 Wasted Turd Penalty

A turd is penalized if it contributes nothing:

Surrounded entirely by unreachable squares

Blocks space already fully owned

Trapped inside self-contained walls

Or reinforces a region with redundant coverage

Cheap approximate tests avoid expensive recomputation.

### 4.3 Blocks (Successful Turd Usage)

Reward “hard territory” created by turds:

Squares opponent can no longer reach

Reductions in opponent Voronoi region

Creation of cutoff regions

Blocking is essentially “permanent Voronoi.”

## 5. Dynamic Weighting

Game phase changes which heuristics matter.

### Early Game

Space > Material

Expansion and frontier contests matter most

Avoid early corner-egg greed (common MAX bot weakness)

### Midgame

Contested region significance dominates

Turd efficiency becomes relevant

Trapdoor risk sharply increases impact

### Late Game

Material > Space

Distance-to-egg bonus active

Endgame egg races decided mostly by distances & remaining moves

### Transitions

Weight transitions depend on:

Turn number

Openness

Fragmentation

Remaining eggs
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

### 7. Final Evaluation Formula (Conceptual)

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
→ Voronoi boundaries jump under small positional changes.
