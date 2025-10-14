def transform(grid: list[list[int]]) -> list[list[int]]:
    if not grid or not grid[0]:
        return []
    rows = len(grid)
    cols = len(grid[0])
    output = [row[:] for row in grid]
    
    for start_row in range(rows):
        for end_row in range(start_row + 1, rows):
            # For this slab start_row to end_row
            is_empty_col = [True] * cols
            for c in range(cols):
                for r in range(start_row, end_row + 1):
                    if grid[r][c] != 0:
                        is_empty_col[c] = False
                        break
            
            # Now find maximal runs of True in is_empty_col
            c = 0
            while c < cols:
                if not is_empty_col[c]:
                    c += 1
                    continue
                # Start of run
                run_start = c
                while c < cols and is_empty_col[c]:
                    c += 1
                run_end = c - 1  # inclusive
                width = run_end - run_start + 1
                if width >= 3:
                    # Fill the rectangle
                    for r in range(start_row, end_row + 1):
                        for cc in range(run_start, run_end + 1):
                            output[r][cc] = 6
                # c is already at next
    return output