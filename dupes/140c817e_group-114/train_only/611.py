def transform(grid: list[list[int]]) -> list[list[int]]:
    if not grid or not grid[0]:
        return []
    h = len(grid)
    w = len(grid[0])
    # Count frequencies to find D
    count = [0] * 10
    for i in range(h):
        for j in range(w):
            c = grid[i][j]
            count[c] += 1
    max_count = max(count)
    if max_count == 0:
        return [row[:] for row in grid]
    D = count.index(max_count)
    if D == 1:
        return [row[:] for row in grid]
    # Find seeds: positions where grid[i][j] == 1
    seeds = []
    for i in range(h):
        for j in range(w):
            if grid[i][j] == 1:
                seeds.append((i, j))
    if not seeds:
        return [row[:] for row in grid]
    # Get unique seed rows and cols
    seed_rows = set(r for r, c in seeds)
    seed_cols = set(c for r, c in seeds)
    # Create output copy
    out = [row[:] for row in grid]
    # Set all positions on seed rows or seed cols to 1
    for i in range(h):
        if i in seed_rows:
            for j in range(w):
                out[i][j] = 1
    for j in range(w):
        if j in seed_cols:
            for i in range(h):
                out[i][j] = 1
    # Set seeds to 2
    for r, c in seeds:
        out[r][c] = 2
    # Now set 3's for adjacent positions to each seed
    directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    for r, c in seeds:
        for dr, dc in directions:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < h and 0 <= nc < w and out[nr][nc] == D:
                out[nr][nc] = 3
    return out