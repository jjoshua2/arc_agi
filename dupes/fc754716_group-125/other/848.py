def transform(grid: list[list[int]]) -> list[list[int]]:
    if not grid or not grid[0]:
        return []
    h = len(grid)
    w = len(grid[0])
    # Find the color C (the only non-zero)
    C = 0
    for i in range(h):
        for j in range(w):
            if grid[i][j] != 0:
                C = grid[i][j]
                break
        if C != 0:
            break
    # If all zero, return zeros
    out = [[0] * w for _ in range(h)]
    if C == 0:
        return out
    # Fill top row
    for j in range(w):
        out[0][j] = C
    # Fill bottom row
    for j in range(w):
        out[h-1][j] = C
    # Fill left column (excluding top/bottom)
    for i in range(1, h-1):
        out[i][0] = C
    # Fill right column (excluding top/bottom)
    for i in range(1, h-1):
        out[i][w-1] = C
    return out