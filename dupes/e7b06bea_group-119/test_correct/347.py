def transform(grid: list[list[int]]) -> list[list[int]]:
    if not grid or not grid[0]:
        return []
    rows = len(grid)
    cols = len(grid[0])
    output = [row[:] for row in grid]
    
    # Find h: number of leading 5's in column 0
    h = 0
    for r in range(rows):
        if grid[r][0] == 5:
            h += 1
        else:
            break
    if h == 0:
        return output
    
    # Find the starting column of the right bars (first non-all-zero column after 0)
    right_start = -1
    for c in range(1, cols):
        all_zero = all(grid[r][c] == 0 for r in range(rows))
        if not all_zero:
            right_start = c
            break
    if right_start == -1:
        return output
    
    # Collect colors from uniform non-zero, non-5 columns starting from right_start
    color_list = []
    for c in range(right_start, cols):
        colors_in_col = set(grid[r][c] for r in range(rows))
        if len(colors_in_col) == 1:
            color = next(iter(colors_in_col))
            if color != 0 and color != 5:
                color_list.append(color)
                # Set this column to 0 in output
                for r in range(rows):
                    output[r][c] = 0
            else:
                # If 0 or 5, skip, but in examples not the case
                pass
        else:
            # Not uniform, skip (but in examples they are)
            pass
    
    m = len(color_list)
    if m == 0:
        return output
    
    # Target column: right_start - 1 (last empty before right bars)
    target_col = right_start - 1
    if target_col < 1:
        return output  # Invalid, but shouldn't happen
    
    # Fill the target column with the cycling pattern
    for r in range(rows):
        block = r // h
        idx = block % m
        output[r][target_col] = color_list[idx]
    
    return output