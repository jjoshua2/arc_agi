def transform(grid: list[list[int]]) -> list[list[int]]:
    h = len(grid)
    if h == 0:
        return []
    w = len(grid[0])
    
    # Find positions of 8's
    eight_positions = [(r, c) for r in range(h) for c in range(w) if grid[r][c] == 8]
    if not eight_positions:
        return [row[:] for row in grid]  # No change if no 8's
    
    min_r = min(pos[0] for pos in eight_positions)
    max_r = max(pos[0] for pos in eight_positions)
    min_c = min(pos[1] for pos in eight_positions)
    max_c = max(pos[1] for pos in eight_positions)
    
    # Find unique C in the bounding box
    colors = set()
    for r in range(min_r, max_r + 1):
        for c in range(min_c, max_c + 1):
            val = grid[r][c]
            if val != 0 and val != 8:
                colors.add(val)
    if not colors:
        C = 0  # Arbitrary, but assume there is
    else:
        C = min(colors)  # If multiple, take min; but examples have one
    
    # Copy grid
    output = [row[:] for row in grid]
    
    # Set borders to 8
    for c in range(min_c, max_c + 1):
        output[min_r][c] = 8
        output[max_r][c] = 8
    for r in range(min_r, max_r + 1):
        output[r][min_c] = 8
        output[r][max_c] = 8
    
    # Fill strict interior with C
    for r in range(min_r + 1, max_r):
        for c in range(min_c + 1, max_c):
            output[r][c] = C
    
    # Fill stems' 0's with C
    # Upper stem
    for r in range(min_r):
        for c in range(min_c, max_c + 1):
            if output[r][c] == 0:
                output[r][c] = C
    # Lower stem
    for r in range(max_r + 1, h):
        for c in range(min_c, max_c + 1):
            if output[r][c] == 0:
                output[r][c] = C
    
    # Fill sides in frame rows' 0's with C
    for r in range(min_r, max_r + 1):
        for c in range(min_c):
            if output[r][c] == 0:
                output[r][c] = C
        for c in range(max_c + 1, w):
            if output[r][c] == 0:
                output[r][c] = C
    
    # Horizontal extensions in frame rows
    for r in range(min_r, max_r + 1):
        # Left side
        left_pos = [(c, grid[r][c]) for c in range(min_c) if grid[r][c] != 0 and grid[r][c] != 8]
        left_pos.sort()
        for i in range(len(left_pos)):
            this_c, this_D = left_pos[i]
            start_c = 0 if i == 0 else left_pos[i - 1][0] + 1
            for k in range(start_c, this_c + 1):
                output[r][k] = this_D
        # Right side
        right_pos = [(c, grid[r][c]) for c in range(max_c + 1, w) if grid[r][c] != 0 and grid[r][c] != 8]
        right_pos.sort()
        for i in range(len(right_pos) - 1, -1, -1):
            this_c, this_D = right_pos[i]
            end_c = w - 1 if i == len(right_pos) - 1 else right_pos[i + 1][0] - 1
            for k in range(this_c, end_c + 1):
                output[r][k] = this_D
    
    # Vertical extensions in stems
    # Upper stems
    for cc in range(min_c, max_c + 1):
        upper_pos = [(rr, grid[rr][cc]) for rr in range(min_r) if grid[rr][cc] != 0 and grid[rr][cc] != 8]
        upper_pos.sort()
        for i in range(len(upper_pos)):
            this_rr, this_D = upper_pos[i]
            start_rr = 0 if i == 0 else upper_pos[i - 1][0] + 1
            for k in range(start_rr, this_rr + 1):
                output[k][cc] = this_D
    # Lower stems
    for cc in range(min_c, max_c + 1):
        lower_pos = [(rr, grid[rr][cc]) for rr in range(max_r + 1, h) if grid[rr][cc] != 0 and grid[rr][cc] != 8]
        lower_pos.sort()
        for i in range(len(lower_pos) - 1, -1, -1):
            this_rr, this_D = lower_pos[i]
            end_rr = h - 1 if i == len(lower_pos) - 1 else lower_pos[i + 1][0] - 1
            for k in range(this_rr, end_rr + 1):
                output[k][cc] = this_D
    
    return output