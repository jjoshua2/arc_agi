def transform(grid: list[list[int]]) -> list[list[int]]:
    if not grid:
        return []
    h = len(grid)
    if h == 0:
        return []
    w = len(grid[0])
    # Find bottom_row: max r with any non-zero in row r
    bottom_row = -1
    for r in range(h):
        if any(cell != 0 for cell in grid[r]):
            bottom_row = r
    if bottom_row == -1:
        return [row[:] for row in grid]
    # Create output all 0
    output = [[0] * w for _ in range(h)]
    # Now, for each r <= bottom_row, for each c with grid[r][c] !=0
    for r in range(bottom_row + 1):
        shift = bottom_row - r
        for c in range(w):
            if grid[r][c] != 0:
                nc = c - shift
                if 0 <= nc < w:
                    output[r][nc] = grid[r][c]
    return output