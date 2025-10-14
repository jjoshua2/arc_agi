def transform(grid: list[list[int]]) -> list[list[int]]:
    mapping = {1:5, 5:1, 2:6, 6:2, 3:4, 4:3, 8:9, 9:8}
    output = [[0] * len(grid[0]) for _ in range(len(grid))]
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            val = grid[i][j]
            output[i][j] = mapping.get(val, val)
    return output