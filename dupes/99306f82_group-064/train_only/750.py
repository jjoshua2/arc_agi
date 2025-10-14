def transform(grid: list[list[int]]) -> list[list[int]]:
    if not grid or not grid[0]:
        return []
    rows = len(grid)
    cols = len(grid[0])
    # Find m: number of seed colors
    m = 0
    min_dim = min(rows, cols)
    while m < min_dim and grid[m][m] != 1:
        m += 1
    if m == 0:
        return [row[:] for row in grid]
    # Extract colors: outermost to innermost
    colors = [grid[i][i] for i in range(m)]
    # Frame starts at row m, col m
    frame_top = m
    frame_left = m
    # Find frame_right from top frame row
    right = frame_left
    while right < cols and grid[frame_top][right] == 1:
        right += 1
    frame_right = right - 1
    # Inside size
    s = frame_right - frame_left - 1
    if s <= 0:
        return [row[:] for row in grid]
    # Inside bounds
    inside_top = frame_top + 1
    inside_left = frame_left + 1
    inside_bot = inside_top + s - 1
    inside_rig = inside_left + s - 1
    # Copy grid
    output = [row[:] for row in grid]
    # Current bounds for filling
    cur_top = inside_top
    cur_left = inside_left
    cur_bot = inside_bot
    cur_rig = inside_rig
    # Fill m-1 border layers
    for i in range(m - 1):
        colr = colors[i]
        # Top row
        for c in range(cur_left, cur_rig + 1):
            output[cur_top][c] = colr
        # Bottom row
        for c in range(cur_left, cur_rig + 1):
            output[cur_bot][c] = colr
        # Left column (excluding corners)
        for r in range(cur_top + 1, cur_bot):
            output[r][cur_left] = colr
        # Right column (excluding corners)
        for r in range(cur_top + 1, cur_bot):
            output[r][cur_rig] = colr
        # Shrink
        cur_top += 1
        cur_left += 1
        cur_bot -= 1
        cur_rig -= 1
        # If shrunk to invalid, break (but shouldn't happen)
        if cur_top > cur_bot or cur_left > cur_rig:
            break
    # Fill remaining inner area with last color
    if m > 0 and cur_top <= cur_bot and cur_left <= cur_rig:
        last_colr = colors[m - 1]
        for r in range(cur_top, cur_bot + 1):
            for c in range(cur_left, cur_rig + 1):
                output[r][c] = last_colr
    return output