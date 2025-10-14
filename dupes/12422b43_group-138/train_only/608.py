def transform(grid):
    if not grid:
        return grid
    
    num_rows = len(grid)
    num_cols = len(grid[0])
    
    # Find H: number of initial 5's in column 0
    H = 0
    while H < num_rows and grid[H][0] == 5:
        H += 1
    
    if H == 0:
        return grid  # No pattern, no change
    
    # Extract pattern: list of lists, each inner list is row[1:] for rows 0 to H-1
    pattern = []
    for i in range(H):
        pat_row = grid[i][1:]
        pattern.append(pat_row)
    
    # Find M: maximum row index with any non-0 in columns 1+
    M = -1
    for r in range(num_rows):
        has_non_zero = False
        for j in range(1, num_cols):
            if grid[r][j] != 0:
                has_non_zero = True
                break
        if has_non_zero:
            M = r
    
    if M == -1:
        start_fill = 0
    else:
        start_fill = M + 1
    
    # Now fill from start_fill to num_rows-1, if the row is all 0 in cols 1+
    for r in range(start_fill, num_rows):
        # Check if all 0 in cols 1+
        all_zero = True
        for j in range(1, num_cols):
            if grid[r][j] != 0:
                all_zero = False
                break
        if all_zero:
            # Compute pat_idx = (r - start_fill) % H
            pat_idx = (r - start_fill) % H
            # Set cols 1+ to pattern[pat_idx]
            for j in range(1, num_cols):
                grid[r][j] = pattern[pat_idx][j - 1]
    
    return grid