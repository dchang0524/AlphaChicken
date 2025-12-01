from numba import njit, int32, float64, boolean
import numpy as np

# Constants for grid
DIST_UNREACHABLE = -1
DIST_BLOCKED = -2

OWNER_NONE = 0
OWNER_ME = 1
OWNER_OPP = 2

@njit(fastmath=True)
def numba_bfs(start_x, start_y, obstacles, dim):
    """
    Super-fast BFS using a fixed-size array as a circular queue.
    Returns the distance matrix.
    """
    dist = np.full((dim, dim), DIST_UNREACHABLE, dtype=np.int32)
    
    # 1. Mark obstacles
    # obstacles grid: 0 = free, 1 = blocked
    for r in range(dim):
        for c in range(dim):
            if obstacles[r, c] == 1:
                dist[r, c] = DIST_BLOCKED

    # Check start
    if dist[start_x, start_y] == DIST_BLOCKED:
        return dist

    # 2. Queue setup (fixed size avoids allocation)
    queue_x = np.empty(dim * dim, dtype=np.int32)
    queue_y = np.empty(dim * dim, dtype=np.int32)
    head = 0
    tail = 0

    # Enqueue start
    dist[start_x, start_y] = 0
    queue_x[tail] = start_x
    queue_y[tail] = start_y
    tail += 1

    # 3. Loop
    while head < tail:
        cx = queue_x[head]
        cy = queue_y[head]
        head += 1
        
        current_dist = dist[cx, cy]
        next_dist = current_dist + 1

        # Neighbors: Up, Right, Down, Left
        # Unrolling loop for speed
        
        # UP
        nx, ny = cx, cy - 1
        if ny >= 0:
            if dist[nx, ny] == DIST_UNREACHABLE:
                dist[nx, ny] = next_dist
                queue_x[tail] = nx
                queue_y[tail] = ny
                tail += 1

        # DOWN
        nx, ny = cx, cy + 1
        if ny < dim:
            if dist[nx, ny] == DIST_UNREACHABLE:
                dist[nx, ny] = next_dist
                queue_x[tail] = nx
                queue_y[tail] = ny
                tail += 1
                
        # LEFT
        nx, ny = cx - 1, cy
        if nx >= 0:
            if dist[nx, ny] == DIST_UNREACHABLE:
                dist[nx, ny] = next_dist
                queue_x[tail] = nx
                queue_y[tail] = ny
                tail += 1

        # RIGHT
        nx, ny = cx + 1, cy
        if nx < dim:
            if dist[nx, ny] == DIST_UNREACHABLE:
                dist[nx, ny] = next_dist
                queue_x[tail] = nx
                queue_y[tail] = ny
                tail += 1

    return dist

@njit(fastmath=True)
def numba_voronoi(dist_me, dist_opp, my_x, my_y, my_even, opp_even, dim):
    """
    Calculates all Voronoi statistics in a single pass.
    """
    my_owned = 0
    opp_owned = 0
    contested = 0
    max_contested_dist = 0
    min_contested_dist = 999
    min_egg_dist = 999
    
    my_voronoi = 0
    opp_voronoi = 0
    
    # Directional counts
    contested_up = 0
    contested_right = 0
    contested_down = 0
    contested_left = 0
    
    # Quadrant counts
    q1 = 0 # Top Right
    q2 = 0 # Top Left
    q3 = 0 # Bottom Left
    q4 = 0 # Bottom Right

    last = dim - 1

    for x in range(dim):
        for y in range(dim):
            d_me = dist_me[x, y]
            d_opp = dist_opp[x, y]
            
            # Reachability
            r_me = (d_me >= 0)
            r_opp = (d_opp >= 0)
            
            owner = OWNER_NONE
            
            # Ownership Logic
            if r_me and (not r_opp or d_me <= d_opp):
                owner = OWNER_ME
                my_owned += 1
            elif r_opp:
                owner = OWNER_OPP
                opp_owned += 1
            
            # Contested Logic
            if r_me and r_opp and owner == OWNER_ME and abs(d_me - d_opp) <= 1:
                contested += 1
                
                if d_me > max_contested_dist:
                    max_contested_dist = d_me
                if d_me < min_contested_dist:
                    min_contested_dist = d_me
                    
                dx = x - my_x
                dy = y - my_y
                
                if dx != 0 or dy != 0:
                    if dx > 0: contested_right += 1
                    else: contested_left += 1 # Technically checks dx <= 0 but 0 is handled by logic flow
                    
                    if dy > 0: contested_down += 1
                    elif dy < 0: contested_up += 1
                    
                    # Quadrants
                    if dx > 0 and dy < 0: q1 += 1
                    elif dx < 0 and dy < 0: q2 += 1
                    elif dx < 0 and dy > 0: q3 += 1
                    elif dx > 0 and dy > 0: q4 += 1

            # Voronoi / Parity Logic
            weight = 1
            if (x == 0 or x == last) and (y == 0 or y == last):
                weight = 3
                
            square_even = ((x + y) & 1) == 0
            
            if square_even == (my_even == 1):
                # Is my parity
                if owner == OWNER_ME:
                    my_voronoi += weight
                    if d_me < min_egg_dist:
                        min_egg_dist = d_me
            elif square_even == (opp_even == 1):
                # Is opp parity
                if owner == OWNER_OPP:
                    opp_voronoi += weight

    return (
        my_owned, opp_owned, contested, 
        max_contested_dist, min_contested_dist, min_egg_dist,
        my_voronoi, opp_voronoi,
        contested_up, contested_right, contested_down, contested_left,
        q1, q2, q3, q4
    )

@njit(fastmath=True)
def numba_evaluate(
    my_eggs, opp_eggs, 
    moves_left, total_moves, 
    vor_score, contested_count, 
    frag_score, max_contested_dist, 
    min_egg_dist, turds_left
):
    """
    Scalar evaluation of the board state using raw numbers.
    Replaces heuristics.evaluate.
    """
    # --- Material Term ---
    mat_diff = my_eggs - opp_eggs
    
    # Avoid division by zero if total_moves is somehow 0 (safety)
    if total_moves > 0:
        phase_mat = max(0.0, min(1.0, moves_left / total_moves))
    else:
        phase_mat = 0.0

    W_MAT_MIN = 5.0
    W_MAT_MAX = 25.0
    
    # --- Space Term ---
    max_contested = 8.0
    openness = 0.0
    if max_contested > 0:
        openness = max(0.0, min(1.0, contested_count / max_contested))

    W_SPACE_MIN = 5.0
    W_SPACE_MAX = 30.0
    
    # --- Other Weights ---
    W_FRAG = 3.0
    W_FRONTIER_DIST = 1.5
    FRONTIER_COEFF = 0.5
    
    # --- Endgame Panic Logic ---
    PANIC_THRESHOLD = 8
    CLOSEST_EGG_COEFF = 0.0
    
    if moves_left <= PANIC_THRESHOLD:
        W_MAT_MIN = 200.0
        W_MAT_MAX = 200.0
        W_SPACE_MIN = 0.5
        W_SPACE_MAX = 0.5
        W_FRAG = 0.0
        W_FRONTIER_DIST = 0.0
        FRONTIER_COEFF = 0.0
        
        if openness > 0:
             CLOSEST_EGG_COEFF = (W_MAT_MAX - 5) * 0.1

    # Interpolate
    w_space = W_SPACE_MIN + (1.0 - openness) * (W_SPACE_MAX - W_SPACE_MIN)
    if openness == 0.0:
        w_space = 0.0
        
    w_mat = W_MAT_MIN + (1.0 - phase_mat) * (W_MAT_MAX - W_MAT_MIN)

    # Calculate Terms
    space_term = w_space * vor_score
    mat_term = w_mat * mat_diff
    
    frag_clamped = max(0.0, min(1.0, frag_score))
    frag_term = -W_FRAG * openness * frag_clamped
    
    # Frontier distance term
    frontier_dist_term = -FRONTIER_COEFF * (1.0 - frag_clamped) * max_contested_dist
    
    # Egg distance penalty (if near end or closed board)
    egg_term = 0.0
    # Assuming 64+ is "infinite/invalid" distance in a small grid
    if min_egg_dist < 64: 
        egg_term = -CLOSEST_EGG_COEFF * min_egg_dist * 0.25

    # Turd Tie-breaker (slight preference to keep turds)
    turd_term = 0.1 * turds_left

    return space_term + mat_term + frag_term + frontier_dist_term + egg_term + turd_term