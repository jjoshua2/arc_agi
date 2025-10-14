def transform(grid: list[list[int]]) -> list[list[int]]:
    if not grid:
        return []
    rows = len(grid)
    cols = len(grid[0])
    output = [row[:] for row in grid]
    for r in range(rows - 1):
        right_color = grid[r][cols - 1]
        for c in range(cols - 1):
            if grid[rows - 1][c] == right_color:
                output[r][c] = right_color
    return output