def transform(grid: list[list[int]]) -> list[list[int]]:
    if not grid or not grid[0]:
        return [row[:] for row in grid]
    rows = len(grid)
    cols = len(grid[0])
    output = [row[:] for row in grid]
    for r in range(1, rows):
        c = 0
        while c < cols:
            if grid[r][c] != 0:
                c += 1
                continue
            # start of potential run
            left = c
            while c < cols and grid[r][c] == 0:
                c += 1
            right = c - 1
            # check if bounded
            left_bound = (left == 0 or grid[r][left - 1] != 0)
            right_bound = (right == cols - 1 or grid[r][right + 1] != 0)
            if not (left_bound and right_bound):
                continue
            # check above all 5
            above_all_5 = True
            for cc in range(left, right + 1):
                if grid[r - 1][cc] != 5:
                    above_all_5 = False
                    break
            if not above_all_5:
                continue
            # now compute maximal h downward
            curr_r = r
            h = 0
            while curr_r < rows:
                # check all 0 in range
                all_zero = True
                for cc in range(left, right + 1):
                    if grid[curr_r][cc] != 0:
                        all_zero = False
                        break
                if not all_zero:
                    break
                # check bounded
                l_b = (left == 0 or grid[curr_r][left - 1] != 0)
                r_b = (right == cols - 1 or grid[curr_r][right + 1] != 0)
                if not (l_b and r_b):
                    break
                h += 1
                curr_r += 1
            # now check if closed below
            closed_below = False
            if curr_r < rows:
                all_5_below = True
                for cc in range(left, right + 1):
                    if grid[curr_r][cc] != 5:
                        all_5_below = False
                        break
                closed_below = all_5_below
            if h > 0 and closed_below:
                # fill the rectangle
                for rr in range(r, r + h):
                    for cc in range(left, right + 1):
                        output[rr][cc] = 4
    return output