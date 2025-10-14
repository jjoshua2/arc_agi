def transform(grid_lst):
    grid = [row[:] for row in grid_lst]
    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0
    
    for r in range(rows):
        # Find segments in this row
        segments = []
        start = -1
        for c in range(cols):
            if grid[r][c] != 0:
                if start == -1:
                    start = c
            else:
                if start != -1:
                    segments.append((start, c - 1))
                    start = -1
        if start != -1:
            segments.append((start, cols - 1))
        
        if len(segments) >= 2:
            # Swap first element of first segment with last element of last segment
            first_start, first_end = segments[0]
            last_start, last_end = segments[-1]
            
            if grid[r][first_start] != grid[r][last_end]:
                grid[r][first_start], grid[r][last_end] = grid[r][last_end], grid[r][first_start]
        
        # For middle segments, swap first and last if different
        for i in range(1, len(segments) - 1):
            seg_start, seg_end = segments[i]
            if grid[r][seg_start] != grid[r][seg_end]:
                grid[r][seg_start], grid[r][seg_end] = grid[r][seg_end], grid[r][seg_start]
    
    return grid