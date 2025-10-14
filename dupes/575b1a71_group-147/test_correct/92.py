def transform(grid: list[list[int]]) -> list[list[int]]:
    if not grid or not grid[0]:
        return grid
    rows = len(grid)
    cols = len(grid[0])
    # Find unique columns with at least one 0
    zero_columns = set()
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == 0:
                zero_columns.add(j)
    if not zero_columns:
        return [row[:] for row in grid]
    # Sort the columns
    sorted_columns = sorted(zero_columns)
    # Create a mapping from column to color
    col_to_color = {sorted_columns[k]: k + 1 for k in range(len(sorted_columns))}
    # Create output grid
    output = [row[:] for row in grid]
    # Assign colors to 0 cells
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == 0:
                output[i][j] = col_to_color[j]
    return output