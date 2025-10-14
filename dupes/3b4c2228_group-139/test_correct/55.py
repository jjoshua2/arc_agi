def transform(grid: list[list[int]]) -> list[list[int]]:
    if not grid or not grid[0]:
        return [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    rows = len(grid)
    cols = len(grid[0])
    
    # Check for special split
    has_special_split = False
    for i in range(1, rows - 1):
        # Check if row i is all 0
        if any(grid[i][j] != 0 for j in range(cols)):
            continue
        # Check has non-0 above
        has_above_non0 = any(any(grid[r][c] != 0 for c in range(cols)) for r in range(i))
        if not has_above_non0:
            continue
        # Check has non-0 below
        has_below_non0 = any(any(grid[r][c] != 0 for c in range(cols)) for r in range(i + 1, rows))
        if not has_below_non0:
            continue
        # Collect colors upper
        colors_upper = set()
        for r in range(i):
            for c in range(cols):
                if grid[r][c] != 0:
                    colors_upper.add(grid[r][c])
        upper_both = 2 in colors_upper and 3 in colors_upper
        # Collect colors lower
        colors_lower = set()
        for r in range(i + 1, rows):
            for c in range(cols):
                if grid[r][c] != 0:
                    colors_lower.add(grid[r][c])
        lower_both = 2 in colors_lower and 3 in colors_lower
        if upper_both and lower_both:
            has_special_split = True
            break
    
    if has_special_split:
        k = 3
    else:
        # Check for adjacency between 2 and 3
        has_adj = False
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == 2:
                    for dr, dc in directions:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 3:
                            has_adj = True
                            break
                    if has_adj:
                        break
                elif grid[r][c] == 3:
                    for dr, dc in directions:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 2:
                            has_adj = True
                            break
                    if has_adj:
                        break
            if has_adj:
                break
        k = 2 if has_adj else 1
    
    # Create 3x3 output
    output = [[0, 0, 0] for _ in range(3)]
    for i in range(k):
        output[i][i] = 1
    return output