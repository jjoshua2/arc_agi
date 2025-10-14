def transform(grid: list[list[int]]) -> list[list[int]]:
    if not grid or len(grid) != 12 or len(grid[0]) != 4:
        raise ValueError("Input must be 12x4 grid")
    output = []
    for i in range(6):
        row = []
        for j in range(4):
            upper = grid[i][j]
            lower = grid[i + 6][j]
            val = 4 if (upper == 3 or lower == 5) else 0
            row.append(val)
        output.append(row)
    return output