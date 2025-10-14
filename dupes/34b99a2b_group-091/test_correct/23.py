def transform(grid: list[list[int]]) -> list[list[int]]:
    rows = len(grid)
    cols = len(grid[0])
    output = []
    for i in range(rows):
        left = [1 if grid[i][j] == 8 else 0 for j in range(4)]
        right = [1 if grid[i][j] == 5 else 0 for j in range(5, 9)]
        out_row = [2 if left[k] != right[k] else 0 for k in range(4)]
        output.append(out_row)
    return output