def transform(grid: list[list[int]]) -> list[list[int]]:
    if not grid or not grid[0]:
        return []
    out = [[0 for _ in range(5)] for _ in range(5)]
    for r in range(5):
        for c in range(5):
            if grid[r][c + 5] == 3:
                out[r][c] = 3
            elif grid[r + 5][c] == 9:
                out[r][c] = 9
            elif grid[r + 5][c + 5] == 8:
                out[r][c] = 8
            elif grid[r][c] == 4:
                out[r][c] = 4
            # else 0, already set
    return out