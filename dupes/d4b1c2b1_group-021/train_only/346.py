def transform(grid: list[list[int]]) -> list[list[int]]:
    if not grid or not grid[0]:
        return []
    rows = len(grid)
    cols = len(grid[0])
    output = [row[:] for row in grid]
    
    # Find all 1 positions
    ones = [(i, j) for i in range(rows) for j in range(cols) if grid[i][j] == 1]
    
    # Find horizontal blocks for each row
    h_blocks = [[] for _ in range(rows)]
    for r in range(rows):
        i = 0
        while i < cols:
            if grid[r][i] == 2:
                left = i
                while i < cols and grid[r][i] == 2:
                    i += 1
                right = i - 1
                h_blocks[r].append((left, right))
            else:
                i += 1
    
    # Find vertical blocks for each column
    v_blocks = [[] for _ in range(cols)]
    for cc in range(cols):
        i = 0
        while i < rows:
            if grid[i][cc] == 2:
                top = i
                while i < rows and grid[i][cc] == 2:
                    i += 1
                bottom = i - 1
                v_blocks[cc].append((top, bottom))
            else:
                i += 1
    
    # Set all 1s to 0
    for r, c in ones:
        output[r][c] = 0
    
    # For each 1, compute target and set to 1
    for r, c in ones:
        # Horizontal
        min_dist_h = float('inf')
        target_h = None
        for left, right in h_blocks[r]:
            if c < left:
                tcol = left - 1
                if tcol >= 0:
                    dist = left - c
                    if dist < min_dist_h:
                        min_dist_h = dist
                        target_h = (r, tcol)
            elif c > right:
                tcol = right + 1
                if tcol < cols:
                    dist = c - right
                    if dist < min_dist_h:
                        min_dist_h = dist
                        target_h = (r, tcol)
        
        # Vertical
        min_dist_v = float('inf')
        target_v = None
        for top, bottom in v_blocks[c]:
            if r < top:
                trow = top - 1
                if trow >= 0:
                    dist = top - r
                    if dist < min_dist_v:
                        min_dist_v = dist
                        target_v = (trow, c)
            elif r > bottom:
                trow = bottom + 1
                if trow < rows:
                    dist = r - bottom
                    if dist < min_dist_v:
                        min_dist_v = dist
                        target_v = (trow, c)
        
        # Decide target
        if target_h is not None and (target_v is None or min_dist_h <= min_dist_v):
            tr, tc = target_h
        elif target_v is not None:
            tr, tc = target_v
        else:
            tr, tc = r, c
        
        output[tr][tc] = 1
    
    return output