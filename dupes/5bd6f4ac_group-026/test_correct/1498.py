def transform(grid: list[list[int]]) -> list[list[int]]:
    # Extract the top-right 3x3 corner from the 9x9 grid
    return [row[6:9] for row in grid[0:3]]