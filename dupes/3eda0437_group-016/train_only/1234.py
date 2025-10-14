def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    if not grid_lst or len(grid_lst) < 2:
        return [row[:] for row in grid_lst]
    rows = len(grid_lst)
    cols = len(grid_lst[0])
    output = [row[:] for row in grid_lst]
    
    for i in range(rows):
        for j in range(i + 1, rows):
            # Determine columns where all cells from row i to j are 0
            full_zero_cols = [False] * cols
            for c in range(cols):
                is_full_zero = all(grid_lst[r][c] == 0 for r in range(i, j + 1))
                full_zero_cols[c] = is_full_zero
            
            # Find maximal runs of such columns
            run_start = -1
            for c in range(cols + 1):
                if c < cols and full_zero_cols[c]:
                    if run_start == -1:
                        run_start = c
                else:
                    if run_start != -1:
                        run_length = c - run_start
                        if run_length >= 3:
                            # Fill the rectangle in output
                            for rr in range(i, j + 1):
                                for cc in range(run_start, c):
                                    output[rr][cc] = 6
                        run_start = -1
    
    return output