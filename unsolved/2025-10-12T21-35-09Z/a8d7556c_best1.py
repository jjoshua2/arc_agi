import numpy as np

def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    grid = np.array(grid_lst)
    rows, cols = grid.shape
    if rows == 0 or cols == 0:
        return grid_lst

    max_hs = [0] * (cols - 1)
    run_infos = [None] * (cols - 1)  # stores (start_row, h) for the max run

    for cc in range(cols - 1):
        current_max_h = 0
        current_start = -1
        r = 0
        while r < rows:
            if grid[r, cc] == 0 and grid[r, cc + 1] == 0:
                start = r
                while r < rows and grid[r, cc] == 0 and grid[r, cc + 1] == 0:
                    r += 1
                h = r - start
                if h > current_max_h:
                    current_max_h = h
                    current_start = start
            else:
                r += 1
        max_hs[cc] = current_max_h
        if current_max_h >= 2:
            run_infos[cc] = (current_start, current_max_h)

    for cc in range(cols - 1):
        h = max_hs[cc]
        if h >= 2:
            left_h = max_hs[cc - 1] if cc > 0 else 0
            right_h = max_hs[cc + 1] if cc < cols - 2 else 0
            if h >= left_h and h >= right_h:
                if run_infos[cc]:
                    start, run_h = run_infos[cc]
                    for rr in range(start, start + run_h):
                        grid[rr, cc] = 2
                        grid[rr, cc + 1] = 2

    return grid.tolist()