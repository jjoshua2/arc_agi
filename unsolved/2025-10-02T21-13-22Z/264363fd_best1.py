def transform(grid: list[list[int]]) -> list[list[int]]:
    if not grid or not grid[0]:
        return []
    h = len(grid)
    w = len(grid[0])
    output = [row[:] for row in grid]
    reds = [(i, j) for i in range(h) for j in range(w) if grid[i][j] == 2]
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    # First pass: set projections to 2 and fill supports with 8
    for red_r, red_c in reds:
        for dr, dc in directions:
            cur_r = red_r
            cur_c = red_c
            while True:
                cur_r += dr
                cur_c += dc
                if not (0 <= cur_r < h and 0 <= cur_c < w):
                    break
                if grid[cur_r][cur_c] == 8:
                    # Hit: set projection to 2
                    output[cur_r][cur_c] = 2
                    # Fill 3x3 support except center
                    for ddr in range(-1, 2):
                        for ddc in range(-1, 2):
                            nr = cur_r + ddr
                            nc = cur_c + ddc
                            if 0 <= nr < h and 0 <= nc < w and not (ddr == 0 and ddc == 0):
                                output[nr][nc] = 8
                    break  # Stop after first hit in this direction
    
    # Second pass: fill paths with 2 only if 0 (respects supports)
    for red_r, red_c in reds:
        for dr, dc in directions:
            path = []
            cur_r = red_r
            cur_c = red_c
            hit = False
            while True:
                cur_r += dr
                cur_c += dc
                if not (0 <= cur_r < h and 0 <= cur_c < w):
                    break
                if grid[cur_r][cur_c] == 8:
                    hit = True
                    break
                path.append((cur_r, cur_c))
            if hit:
                for pr, pc in path:
                    if output[pr][pc] == 0:
                        output[pr][pc] = 2
    
    return output