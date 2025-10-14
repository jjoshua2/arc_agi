def transform(input_grid: list[list[int]]) -> list[list[int]]:
    if not input_grid:
        return []
    grid = [row[:] for row in input_grid]
    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0
    for r in range(rows):
        max_c = -1
        for c in range(cols):
            if grid[r][c] != 0:
                max_c = c
        if max_c == -1:
            continue
        seed_len = max_c + 1
        p = seed_len
        for possible_p in range(1, seed_len):
            is_periodic = True
            for i in range(seed_len - possible_p):
                if grid[r][i] != grid[r][i + possible_p]:
                    is_periodic = False
                    break
            if is_periodic:
                p = possible_p
                break
        tile = grid[r][:p]
        for c in range(cols):
            grid[r][c] = tile[c % p]
    return grid