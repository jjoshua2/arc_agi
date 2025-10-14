def transform(grid: list[list[int]]) -> list[list[int]]:
    if not grid or not grid[0]:
        return []
    rows = len(grid)
    cols = len(grid[0])
    output = [row[:] for row in grid]
    # Horizontal fills: in each row, fill gaps between consecutive 1s
    for r in range(rows):
        positions = [c for c in range(cols) if grid[r][c] == 1]
        for i in range(len(positions) - 1):
            start_c = positions[i] + 1
            end_c = positions[i + 1] - 1
            for c in range(start_c, end_c + 1):
                output[r][c] = 2
    # Vertical fills: in each column, fill gaps between consecutive 1s
    for c in range(cols):
        positions = [r for r in range(rows) if grid[r][c] == 1]
        for i in range(len(positions) - 1):
            start_r = positions[i] + 1
            end_r = positions[i + 1] - 1
            for r in range(start_r, end_r + 1):
                output[r][c] = 2
    return output