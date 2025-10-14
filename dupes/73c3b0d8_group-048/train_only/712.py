def spread(grid, row, col, rows, cols):
    if row < 0 or row >= rows or col < 0 or col >= cols:
        return
    grid[row][col] = 4
    # left arm
    r, c = row - 1, col - 1
    while r >= 0 and c >= 0:
        grid[r][c] = 4
        r -= 1
        c -= 1
    # right arm
    r, c = row - 1, col + 1
    while r >= 0 and c < cols:
        grid[r][c] = 4
        r -= 1
        c += 1

def transform(input_grid: list[list[int]]) -> list[list[int]]:
    if not input_grid or not input_grid[0]:
        return []
    rows = len(input_grid)
    if rows == 0:
        return []
    cols = len(input_grid[0])
    output = [row[:] for row in input_grid]
    # Find red_row
    red_row = -1
    for i in range(rows):
        if all(cell == 2 for cell in input_grid[i]):
            red_row = i
            break
    if red_row == -1:
        return output
    target_row = red_row - 1
    # Process each yellow cell
    for r in range(rows):
        for c in range(cols):
            if input_grid[r][c] == 4:
                output[r][c] = 0  # clear original
                if r > red_row:  # below
                    new_r = r + 1
                    if new_r < rows:
                        output[new_r][c] = 4
                elif r < red_row:  # above
                    new_r = r + 1
                    spread_r = None
                    if new_r == red_row:
                        spread_r = r
                    elif new_r == target_row:
                        spread_r = new_r
                    if spread_r is not None:
                        spread(output, spread_r, c, rows, cols)
                    elif new_r < red_row:
                        output[new_r][c] = 4
    return output