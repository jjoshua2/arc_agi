def transform(grid: list[list[int]]) -> list[list[int]]:
    if not grid or not grid[0]:
        return []
    h = len(grid)
    w = len(grid[0])
    # Step 1: Find bounding box of 5's
    five_min_r = h
    five_max_r = -1
    five_min_c = w
    five_max_c = -1
    for r in range(h):
        for cc in range(w):
            if grid[r][cc] == 5:
                five_min_r = min(five_min_r, r)
                five_max_r = max(five_max_r, r)
                five_min_c = min(five_min_c, cc)
                five_max_c = max(five_max_c, cc)
    if five_min_r > five_max_r:  # No 5's
        return [row[:] for row in grid]
    # Step 2: Copy grid
    output = [row[:] for row in grid]
    # Step 3: For each color c != 5
    for c in range(1, 10):
        if c == 5:
            continue
        seeds = [(r, cc) for r in range(h) for cc in range(w) if grid[r][cc] == c]
        if not seeds:
            continue
        c_min_r = min(r for r, _ in seeds)
        c_max_r = max(r for r, _ in seeds)
        c_min_c = min(cc for _, cc in seeds)
        c_max_c = max(cc for _, cc in seeds)
        # Fill region intersection
        fill_min_r = max(five_min_r, c_min_r)
        fill_max_r = min(five_max_r, c_max_r)
        fill_min_c = max(five_min_c, c_min_c)
        fill_max_c = min(five_max_c, c_max_c)
        if fill_min_r > fill_max_r or fill_min_c > fill_max_c:
            continue
        for i in range(fill_min_r, fill_max_r + 1):
            for j in range(fill_min_c, fill_max_c + 1):
                if grid[i][j] == 5:
                    output[i][j] = c
    # Step 4: Remove external seeds
    for r in range(h):
        for cc in range(w):
            if grid[r][cc] != 0 and grid[r][cc] != 5:
                if r < five_min_r or r > five_max_r or cc < five_min_c or cc > five_max_c:
                    output[r][cc] = 0
    return output