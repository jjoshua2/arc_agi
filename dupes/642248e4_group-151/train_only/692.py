def transform(grid: list[list[int]]) -> list[list[int]]:
    if not grid or not grid[0]:
        return []
    h = len(grid)
    w = len(grid[0])
    output = [row[:] for row in grid]
    
    # Check top uniform
    top_color = None
    if w > 0:
        tc = grid[0][0]
        if all(grid[0][j] == tc for j in range(w)):
            top_color = tc
    
    # Check bottom uniform
    bottom_color = None
    if w > 0:
        bc = grid[h-1][0]
        if all(grid[h-1][j] == bc for j in range(w)):
            bottom_color = bc
    
    # Check left uniform
    left_color = None
    if h > 0:
        lc = grid[0][0]
        if all(grid[i][0] == lc for i in range(h)):
            left_color = lc
    
    # Check right uniform
    right_color = None
    if h > 0 and w > 0:
        rc = grid[0][w-1]
        if all(grid[i][w-1] == rc for i in range(h)):
            right_color = rc
    
    # Process each 1
    for r in range(h):
        for c in range(w):
            if grid[r][c] != 1:
                continue
            
            # Vertical propagation if applicable
            if top_color is not None and bottom_color is not None:
                dist_top = r
                dist_bot = h - 1 - r
                if dist_top < dist_bot:
                    if r > 0 and output[r-1][c] == 0:
                        output[r-1][c] = top_color
                elif dist_bot < dist_top:
                    if r < h - 1 and output[r+1][c] == 0:
                        output[r+1][c] = bottom_color
            
            # Horizontal propagation if applicable
            if left_color is not None and right_color is not None:
                dist_left = c
                dist_right = w - 1 - c
                if dist_left < dist_right:
                    if c > 0 and output[r][c-1] == 0:
                        output[r][c-1] = left_color
                elif dist_right < dist_left:
                    if c < w - 1 and output[r][c+1] == 0:
                        output[r][c+1] = right_color
    
    return output