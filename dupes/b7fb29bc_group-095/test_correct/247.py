def transform(grid: list[list[int]]) -> list[list[int]]:
    if not grid or not grid[0]:
        return []
    rows = len(grid)
    cols = len(grid[0])
    three_pos = [(i, j) for i in range(rows) for j in range(cols) if grid[i][j] == 3]
    if not three_pos:
        return [row[:] for row in grid]
    min_r = min(i for i, j in three_pos)
    max_r = max(i for i, j in three_pos)
    min_c = min(j for i, j in three_pos)
    max_c = max(j for i, j in three_pos)
    seeds = []
    for i, j in three_pos:
        is_perimeter = (i == min_r or i == max_r or
                        (j == min_c and min_r < i < max_r) or
                        (j == max_c and min_r < i < max_r))
        if not is_perimeter:
            seeds.append((i, j))
    output = [row[:] for row in grid]
    for r in range(min_r + 1, max_r):
        for c in range(min_c + 1, max_c):
            if output[r][c] == 0:
                if seeds:
                    min_d = min(max(abs(r - sr), abs(c - sc)) for sr, sc in seeds)
                    color = 2 if min_d % 2 == 0 else 4
                    output[r][c] = color
    return output