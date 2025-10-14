def transform(grid: list[list[int]]) -> list[list[int]]:
    if len(grid) != 3 or len(grid[0]) != 3:
        return []
    # Initialize 9x9 output with zeros
    output = [[0] * 9 for _ in range(9)]
    # Find uniform row (first one)
    uniform_row = None
    for i in range(3):
        if len(set(grid[i])) == 1:
            uniform_row = i
            break
    if uniform_row is not None:
        # Horizontal mode
        start_r = uniform_row * 3
        for ii in range(3):
            tiled_row = grid[ii] + grid[ii] + grid[ii]
            output[start_r + ii] = tiled_row[:9]  # Ensure length 9, though it should be
        return output
    # Find uniform column (first one)
    uniform_col = None
    for j in range(3):
        col = [grid[i][j] for i in range(3)]
        if len(set(col)) == 1:
            uniform_col = j
            break
    if uniform_col is not None:
        # Vertical mode
        start_c = uniform_col * 3
        for rr in range(9):
            input_row = grid[rr % 3]
            for cc in range(3):
                g_c = start_c + cc
                output[rr][g_c] = input_row[cc]
        return output
    # If no uniform row or column, return all zeros?
    return output