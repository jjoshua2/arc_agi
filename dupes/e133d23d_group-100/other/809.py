def transform(grid: list[list[int]]) -> list[list[int]]:
    if not grid or len(grid) != 3 or len(grid[0]) != 7:
        return []  # Invalid input, but assuming always 3x7
    output = [[0 for _ in range(3)] for _ in range(3)]
    for i in range(3):
        for j in range(3):
            left = grid[i][j]
            right = grid[i][j + 4]
            if left != 0 or right != 0:
                output[i][j] = 2
            else:
                output[i][j] = 0
    return output