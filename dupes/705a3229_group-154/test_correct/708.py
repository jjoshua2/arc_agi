def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    if not grid_lst:
        return []
    rows = len(grid_lst)
    if rows == 0:
        return []
    cols = len(grid_lst[0])
    out = [row[:] for row in grid_lst]
    
    for r in range(rows):
        for c in range(cols):
            k = grid_lst[r][c]
            if k == 0:
                continue
            # Vertical fill
            dist_up = r
            dist_down = rows - 1 - r
            if dist_up <= dist_down:
                # Fill up
                for rr in range(r + 1):
                    out[rr][c] = k
            else:
                # Fill down
                for rr in range(r, rows):
                    out[rr][c] = k
            # Horizontal fill
            dist_left = c
            dist_right = cols - 1 - c
            if dist_left <= dist_right:
                # Fill left
                for cc in range(c + 1):
                    out[r][cc] = k
            else:
                # Fill right
                for cc in range(c, cols):
                    out[r][cc] = k
    
    return out