def transform(grid: list[list[int]]) -> list[list[int]]:
    if not grid or not grid[0]:
        return grid
    m = len(grid)
    n = len(grid[0])
    output = [row[:] for row in grid]
    seeds = []
    for r in range(m):
        for c in range(n):
            k = grid[r][c]
            if k != 0:
                seeds.append((r, c, k))
    coverage = [[0] * n for _ in range(m)]
    for r, c, k in seeds:
        if k == 1:
            for dc in [1, 2]:
                nc = c + dc
                if 0 <= nc < n:
                    coverage[r][nc] += 1
        elif k == 2:
            for di in range(1, 4):
                nc = c - di
                if 0 <= nc < n:
                    coverage[r][nc] += 1
        elif k == 3:
            for di in range(1, 3):
                nr = r + di
                if 0 <= nr < m:
                    coverage[nr][c] += 1
        elif k == 6:
            for di in range(1, 6):
                nr = r - di
                if 0 <= nr < m:
                    coverage[nr][c] += 1
    for rr in range(m):
        for cc in range(n):
            if coverage[rr][cc] > 0 and output[rr][cc] == 0:
                if coverage[rr][cc] >= 2:
                    output[rr][cc] = 4
                else:
                    output[rr][cc] = 5
    for r, c, k in seeds:
        if k == 1:
            nr = r - 1
            nc = c + 2
            if 0 <= nr < m and 0 <= nc < n:
                output[nr][nc] = 1
        elif k == 2:
            nc = c - 4
            if 0 <= nc < n:
                output[r][nc] = 2
        elif k == 3:
            nr = r + 3
            if 0 <= nr < m:
                output[nr][c] = 3
        elif k == 6:
            nr = r - 6
            if 0 <= nr < m:
                output[nr][c] = 6
    for rr in range(m):
        for cc in range(n):
            if output[rr][cc] == 4:
                for step in range(1, 5):
                    nr2 = rr + step
                    nc2 = cc - step
                    if 0 <= nr2 < m and 0 <= nc2 < n and output[nr2][nc2] == 0:
                        output[nr2][nc2] = 4
    return output