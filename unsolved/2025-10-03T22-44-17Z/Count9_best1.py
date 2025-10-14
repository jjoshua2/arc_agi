def transform(grid: list[list[int]]) -> list[list[int]]:
    if not grid or not grid[0]:
        return []
    non_zero_colors = set()
    for row in grid:
        for cell in row:
            if cell != 0:
                non_zero_colors.add(cell)
    k = len(non_zero_colors)
    return [[0] * k]