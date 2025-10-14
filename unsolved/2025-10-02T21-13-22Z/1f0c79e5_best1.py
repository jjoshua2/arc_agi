def transform(input_grid: list[list[int]]) -> list[list[int]]:
    # Copy grid
    if not input_grid or not input_grid[0]:
        return [row[:] for row in input_grid]
    rows = len(input_grid)
    cols = len(input_grid[0])
    grid = [row[:] for row in input_grid]
    
    # Find the 2x2 window that contains the most non-zero cells (robust way to locate the colored block)
    best_count = -1
    best_r = 0
    best_c = 0
    for r in range(rows - 1):
        for c in range(cols - 1):
            cnt = 0
            if grid[r][c] != 0: cnt += 1
            if grid[r][c+1] != 0: cnt += 1
            if grid[r+1][c] != 0: cnt += 1
            if grid[r+1][c+1] != 0: cnt += 1
            if cnt > best_count:
                best_count = cnt
                best_r = r
                best_c = c
    
    # If no non-zero found (edge case), return unchanged
    if best_count <= 0:
        return grid
    
    r0 = best_r
    c0 = best_c
    A = grid[r0][c0]
    B = grid[r0][c0+1]
    C = grid[r0+1][c0]
    D = grid[r0+1][c0+1]
    
    # Choose the color: the maximum numeric color among the four
    chosen = max(A, B, C, D)
    
    # Fill the original 2x2 with chosen color
    grid[r0][c0] = chosen
    grid[r0][c0+1] = chosen
    grid[r0+1][c0] = chosen
    grid[r0+1][c0+1] = chosen
    
    # Decide direction and whether to propagate only forward or both ways
    # dc is always +1 (we always move rightwards column-wise)
    dc = 1
    if A == chosen and B == chosen:
        dr = 1
        forward_only = True
    elif A == chosen and C == chosen:
        dr = -1
        forward_only = True
    else:
        dr = -1
        forward_only = False
    
    # Compute k bounds so that the 2x2 tile at (r0 + k*dr, c0 + k*dc) fits entirely inside the grid
    # Conditions: 0 <= r0 + k*dr <= rows-2  and  0 <= c0 + k*dc <= cols-2
    if dr == 1:
        # k >= -r0 and k <= rows-2 - r0
        # k >= -c0 and k <= cols-2 - c0
        k_min_allowed = max(-r0, -c0)
        k_max_allowed = min((rows - 2) - r0, (cols - 2) - c0)
    else:  # dr == -1
        # r0 - k >= 0 -> k <= r0
        # r0 - k <= rows-2 -> k >= r0 - (rows-2)
        # c0 + k >= 0 -> k >= -c0
        # c0 + k <= cols-2 -> k <= cols-2 - c0
        k_min_allowed = max(r0 - (rows - 2), -c0)
        k_max_allowed = min(r0, (cols - 2) - c0)
    
    if forward_only:
        k_min = max(0, k_min_allowed)
        k_max = k_max_allowed
    else:
        k_min = k_min_allowed
        k_max = k_max_allowed
    
    # Ensure integer bounds
    k_min = int(k_min)
    k_max = int(k_max)
    
    # For each valid k in range, stamp a 2x2 tile of chosen color
    for k in range(k_min, k_max + 1):
        rr = r0 + k * dr
        cc = c0 + k * dc
        if 0 <= rr <= rows - 2 and 0 <= cc <= cols - 2:
            grid[rr][cc] = chosen
            grid[rr][cc + 1] = chosen
            grid[rr + 1][cc] = chosen
            grid[rr + 1][cc + 1] = chosen
    
    return grid