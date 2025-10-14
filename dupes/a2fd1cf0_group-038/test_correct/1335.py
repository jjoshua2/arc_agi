def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    if not grid_lst or not grid_lst[0]:
        return []
    h, w = len(grid_lst), len(grid_lst[0])
    output = [row[:] for row in grid_lst]
    
    # Find position of 2 and 3
    rx, cx, ry, cy = -1, -1, -1, -1
    for i in range(h):
        for j in range(w):
            if grid_lst[i][j] == 2:
                rx, cx = i, j
            elif grid_lst[i][j] == 3:
                ry, cy = i, j
    
    if rx == -1 or ry == -1:
        return output  # No change if missing
    
    # Horizontal: row rx, cols min(cx,cy) to max(cx,cy), exclude cx
    min_c = min(cx, cy)
    max_c = max(cx, cy)
    for j in range(min_c, max_c + 1):
        if j != cx:
            output[rx][j] = 8
    
    # Vertical: col cy, rows min(rx,ry) to max(rx,ry), exclude ry
    min_r = min(rx, ry)
    max_r = max(rx, ry)
    for i in range(min_r, max_r + 1):
        if i != ry:
            output[i][cy] = 8
    
    return output