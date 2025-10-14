def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    if not grid_lst:
        return []
    grid = [row[:] for row in grid_lst]
    rows = len(grid)
    if rows == 0:
        return grid
    cols = len(grid[0])
    for rs in range(rows):
        for re in range(rs, rows):
            for cs in range(cols):
                for ce in range(cs, cols):
                    # Check all zero inside
                    all_zero = True
                    for i in range(rs, re + 1):
                        for j in range(cs, ce + 1):
                            if grid[i][j] != 0:
                                all_zero = False
                                break
                        if not all_zero:
                            break
                    if not all_zero:
                        continue
                    area = (re - rs + 1) * (ce - cs + 1)
                    if area == 0:
                        continue
                    # Check bounds
                    top_ok = (rs == 0) or all(grid[rs - 1][j] == 2 for j in range(cs, ce + 1))
                    bottom_ok = (re == rows - 1) or all(grid[re + 1][j] == 2 for j in range(cs, ce + 1))
                    left_ok = (cs == 0) or all(grid[i][cs - 1] == 2 for i in range(rs, re + 1))
                    right_ok = (ce == cols - 1) or all(grid[i][ce + 1] == 2 for i in range(rs, re + 1))
                    if not (top_ok and bottom_ok and left_ok and right_ok):
                        continue
                    # Touches border
                    touches_border = (rs == 0 or re == rows - 1 or cs == 0 or ce == cols - 1)
                    if not touches_border:
                        continue
                    # Special case for left-touching
                    touches_left = (cs == 0)
                    if touches_left and not (rs == 0 and area <= 8):
                        continue
                    # Determine color
                    color = 8 if area <= 8 else 1
                    # Fill the rectangle
                    for i in range(rs, re + 1):
                        for j in range(cs, ce + 1):
                            grid[i][j] = color
    return grid