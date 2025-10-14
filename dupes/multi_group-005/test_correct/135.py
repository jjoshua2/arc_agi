def transform(grid: list[list[int]]) -> list[list[int]]:
    if len(grid) != 15 or len(grid[0]) != 5:
        raise ValueError("Input must be 15x5 grid")
    output = [[0] * 5 for _ in range(5)]
    for i in range(5):
        # Place purple (middle block, rows 5-9) first
        for j in range(5):
            if grid[5 + i][j] != 0:
                output[i][j] = grid[5 + i][j]
        # Then place blue (top block, rows 0-4), overriding purple
        for j in range(5):
            if grid[i][j] != 0:
                output[i][j] = grid[i][j]
        # Then place pink (bottom block, rows 10-14), overriding previous
        for j in range(5):
            if grid[10 + i][j] != 0:
                output[i][j] = grid[10 + i][j]
    return output