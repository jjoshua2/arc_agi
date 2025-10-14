def transform(grid: list[list[int]]) -> list[list[int]]:
    if not grid or not grid[0]:
        return [row[:] for row in grid]
    
    rows = len(grid)
    cols = len(grid[0])
    output = [row[:] for row in grid]
    
    for r1 in range(rows):
        for c1 in range(cols):
            if grid[r1][c1] != 0:
                continue
            for r2 in range(r1, rows):
                broke = False
                for c2 in range(c1, cols):
                    # Check if the entire rectangle r1..r2, c1..c2 is all 0s in input
                    all_zero = True
                    for rr in range(r1, r2 + 1):
                        for cc in range(c1, c2 + 1):
                            if grid[rr][cc] != 0:
                                all_zero = False
                                break
                        if not all_zero:
                            break
                    if not all_zero:
                        break  # Larger c2 will also not be all zero
                    
                    # Check boundaries
                    # Top
                    top_ok = (r1 == 0) or all(grid[r1 - 1][cc] == 5 for cc in range(c1, c2 + 1))
                    # Bottom
                    bottom_ok = (r2 == rows - 1) or all(grid[r2 + 1][cc] == 5 for cc in range(c1, c2 + 1))
                    # Left
                    left_ok = (c1 == 0) or all(grid[rr][c1 - 1] == 5 for rr in range(r1, r2 + 1))
                    # Right
                    right_ok = (c2 == cols - 1) or all(grid[rr][c2 + 1] == 5 for rr in range(r1, r2 + 1))
                    
                    if all_zero and top_ok and bottom_ok and left_ok and right_ok:
                        # Fill the rectangle
                        for rr in range(r1, r2 + 1):
                            for cc in range(c1, c2 + 1):
                                output[rr][cc] = 4
                # After c2 loop for this r2
                # To optimize r2: if the current r2 row has a non-zero in col c1 (since we start from c1), but actually, since c2 starts from c1, if grid[r2][c1] !=0, then even 1-wide won't be all zero, but since we continued only if grid[r1][c1]==0, but for larger r2, we check inside.
                # No need for more optimization.
    
    return output