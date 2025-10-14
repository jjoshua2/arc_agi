def transform(grid: list[list[int]]) -> list[list[int]]:
    rows = len(grid)
    cols = len(grid[0])
    # Find the position of 8
    r8 = c8 = -1
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == 8:
                r8, c8 = i, j
                break
        if r8 != -1:
            break
    if r8 == -1:
        return [row[:] for row in grid]
    output = [row[:] for row in grid]
    # Upper part (above the 8, shift right)
    for d in range(1, r8 + 1):
        row = r8 - d
        shift = d - 1
        if d % 2 == 1:  # odd: single
            col = c8 + shift
            if 0 <= col < cols:
                output[row][col] = 5
        else:  # even: triple centered
            center = c8 + shift
            for dc in [-1, 0, 1]:
                col = center + dc
                if 0 <= col < cols:
                    output[row][col] = 5
    # Lower part (below the 8, shift left)
    for d in range(1, rows - r8):
        row = r8 + d
        shift = d - 1
        if d % 2 == 1:  # odd: single
            col = c8 - shift
            if 0 <= col < cols:
                output[row][col] = 5
        else:  # even: triple centered
            center = c8 - shift
            for dc in [-1, 0, 1]:
                col = center + dc
                if 0 <= col < cols:
                    output[row][col] = 5
    return output