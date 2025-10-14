def transform(grid: list[list[int]]) -> list[list[int]]:
    if not grid or not grid[0]:
        return []
    h = len(grid)
    w = len(grid[0])
    seeds = []
    for r in range(h):
        for c in range(w):
            if grid[r][c] != 0:
                seeds.append((r, c, grid[r][c]))
    if not seeds:
        return [row[:] for row in grid]
    min_r = min(r for r, c, col in seeds)
    max_r = max(r for r, c, col in seeds)
    min_c = min(c for r, c, col in seeds)
    max_c = max(c for r, c, col in seeds)
    if min_r == max_r or min_c == max_c:
        output = [[0] * w for _ in range(h)]
        for r, c, col in seeds:
            output[r][c] = col
        return output
    # Find top_color
    top_color = None
    for c in range(w):
        if grid[min_r][c] != 0:
            top_color = grid[min_r][c]
            break
    # Find bottom_color
    bottom_color = None
    for c in range(w):
        if grid[max_r][c] != 0:
            bottom_color = grid[max_r][c]
            break
    # Find left_color
    left_color = None
    for r in range(h):
        if grid[r][min_c] != 0:
            left_color = grid[r][min_c]
            break
    # Find right_color
    right_color = None
    for r in range(h):
        if grid[r][max_c] != 0:
            right_color = grid[r][max_c]
            break
    # Create output
    output = [[0] * w for _ in range(h)]
    # Top horizontal
    for c in range(min_c, max_c + 1):
        output[min_r][c] = top_color
    # Bottom horizontal
    for c in range(min_c, max_c + 1):
        output[max_r][c] = bottom_color
    # Left vertical
    for r in range(min_r + 1, max_r):
        output[r][min_c] = left_color
    # Right vertical
    for r in range(min_r + 1, max_r):
        output[r][max_c] = right_color
    # Internal seeds
    internal_seeds = [(r, c, col) for r, c, col in seeds if min_r < r < max_r and min_c < c < max_c]
    internal_color = 5
    for ir, ic, _ in internal_seeds:
        # Vertical fill
        for rr in range(min_r + 1, max_r):
            if output[rr][ic] == 0:
                output[rr][ic] = internal_color
        # Horizontal fill
        for cc in range(min_c, max_c + 1):
            if output[ir][cc] == 0:
                output[ir][cc] = internal_color
    # Set all seeds
    for r, c, col in seeds:
        output[r][c] = col
    return output