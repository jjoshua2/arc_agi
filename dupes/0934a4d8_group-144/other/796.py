def transform(grid: list[list[int]]) -> list[list[int]]:
    if not grid or len(grid) != 12 or len(grid[0]) != 4:
        return []  # Assuming valid inputs as per examples
    output = [[0] * 4 for _ in range(6)]
    for i in range(6):
        for j in range(4):
            if grid[i][j] != 0 or grid[i + 6][j] != 0:
                output[i][j] = 4
    return output