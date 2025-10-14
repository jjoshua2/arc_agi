def transform(grid: list[list[int]]) -> list[list[int]]:
    if not grid or not grid[0]:
        return grid
    rows = len(grid)
    cols = len(grid[0])
    # Find the seed
    seed_r, seed_c, color = -1, -1, -1
    for r in range(rows):
        for c in range(cols):
            val = grid[r][c]
            if val != 0 and val != 5:
                if seed_r != -1:
                    # Assume only one, but if multiple, this would overwrite
                    pass
                seed_r, seed_c, color = r, c, val
    if seed_r == -1:
        return [row[:] for row in grid]  # No seed, no change
    p = seed_r % 2
    q = seed_c % 2
    out = [row[:] for row in grid]
    for r in range(rows):
        if r % 2 != p:
            continue
        for c in range(cols):
            if c % 2 == q and out[r][c] == 0:
                out[r][c] = color
    return out