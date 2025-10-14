def transform(grid: list[list[int]]) -> list[list[int]]:
    if not grid or not grid[0]:
        return []
    rows = len(grid)
    cols = len(grid[0])
    output = [row[:] for row in grid]
    
    # Find all green frames: (r, c1, c, c2)
    frames = []
    for r in range(1, rows - 1):
        for c in range(1, cols - 1):
            c1 = c - 1
            c2 = c + 1
            if (grid[r][c1] == 3 and grid[r][c] == 2 and grid[r][c2] == 3 and
                all(grid[r - 1][j] == 3 for j in (c1, c, c2)) and
                all(grid[r + 1][j] == 3 for j in (c1, c, c2))):
                frames.append((r, c1, c, c2))
    
    # Process green frames
    for fi, (r, c1, c, c2) in enumerate(frames):
        # Middle row: fill 0s to 1
        for j in range(cols):
            if output[r][j] == 0:
                output[r][j] = 1
        # Pierce other frames' sides in this middle row
        for oi, (or_, oc1, oc, oc2) in enumerate(frames):
            if oi == fi:
                continue
            if oc1 < cols and output[r][oc1] == 2:
                output[r][oc1] = 1
            if oc2 < cols and output[r][oc2] == 2:
                output[r][oc2] = 1
        # Side rows: fill widths of all frames
        for dr in [-1, 1]:
            sr = r + dr
            if 0 <= sr < rows:
                for (_, fc1, fc, fc2) in frames:
                    for j in range(fc1, fc2 + 1):
                        if j < cols and (output[sr][j] == 0 or output[sr][j] == 2):
                            output[sr][j] = 1
    
    # Process horizontal red rows: rows where all cells are 2 in input
    for s in range(rows):
        is_red_row = all(grid[s][j] == 2 for j in range(cols))
        if is_red_row:
            # Pierce all sides in this row
            for (_, fc1, fc, fc2) in frames:
                if fc1 < cols and output[s][fc1] == 2:
                    output[s][fc1] = 1
                if fc2 < cols and output[s][fc2] == 2:
                    output[s][fc2] = 1
            # Side rows: fill widths of all frames
            for dr in [-1, 1]:
                sr = s + dr
                if 0 <= sr < rows:
                    for (_, fc1, fc, fc2) in frames:
                        for j in range(fc1, fc2 + 1):
                            if j < cols and (output[sr][j] == 0 or output[sr][j] == 2):
                                output[sr][j] = 1
    
    # Vertical fills: for each center column, fill 0s to 1
    for (_, c1, c, c2) in frames:
        for i in range(rows):
            if output[i][c] == 0:
                output[i][c] = 1
    
    return output