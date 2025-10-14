def transform(grid: list[list[int]]) -> list[list[int]]:
    if not grid or len(grid) != 3 or len(grid[0]) != 7:
        return []
    out = [[0 for _ in range(3)] for _ in range(3)]
    for i in range(3):
        for j in range(3):
            left = grid[i][j] != 0
            right = grid[i][j + 4] != 0
            out[i][j] = 2 if left or right else 0
    return out