def transform(grid: list[list[int]]) -> list[list[int]]:
    if not grid or not grid[0]:
        return []
    rows = len(grid)
    cols = len(grid[0])
    output = [row[:] for row in grid]
    
    for top in range(rows):
        for bottom in range(top + 1, rows):
            h = bottom - top + 1
            for left in range(cols):
                for right in range(left + 1, cols):
                    w = right - left + 1
                    # Check all inside are 0
                    all_zero_inside = True
                    for r in range(top, bottom + 1):
                        for c in range(left, right + 1):
                            if grid[r][c] != 0:
                                all_zero_inside = False
                                break
                        if not all_zero_inside:
                            break
                    if not all_zero_inside:
                        continue
                    # Check cannot extend up
                    can_extend_up = False
                    if top > 0:
                        all_zero_up = True
                        for c in range(left, right + 1):
                            if grid[top - 1][c] != 0:
                                all_zero_up = False
                                break
                        if all_zero_up:
                            can_extend_up = True
                    # Check cannot extend down
                    can_extend_down = False
                    if bottom < rows - 1:
                        all_zero_down = True
                        for c in range(left, right + 1):
                            if grid[bottom + 1][c] != 0:
                                all_zero_down = False
                                break
                        if all_zero_down:
                            can_extend_down = True
                    # Check cannot extend left
                    can_extend_left = False
                    if left > 0:
                        all_zero_left = True
                        for r in range(top, bottom + 1):
                            if grid[r][left - 1] != 0:
                                all_zero_left = False
                                break
                        if all_zero_left:
                            can_extend_left = True
                    # Check cannot extend right
                    can_extend_right = False
                    if right < cols - 1:
                        all_zero_right = True
                        for r in range(top, bottom + 1):
                            if grid[r][right + 1] != 0:
                                all_zero_right = False
                                break
                        if all_zero_right:
                            can_extend_right = True
                    # If maximal in all directions
                    if (not can_extend_up and not can_extend_down and
                        not can_extend_left and not can_extend_right):
                        # Fill with 2
                        for r in range(top, bottom + 1):
                            for c in range(left, right + 1):
                                output[r][c] = 2
    return output