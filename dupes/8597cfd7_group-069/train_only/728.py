def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    if not grid_lst:
        return []
    h = len(grid_lst)
    if h == 0:
        return []
    w = len(grid_lst[0])
    
    # Find the divider row (all 5's)
    divider = -1
    for r in range(h):
        if all(cell == 5 for cell in grid_lst[r]):
            divider = r
            break
    if divider == -1:
        return [[0]]  # Assume exists, but handle
    
    upper_start = 0
    upper_end = divider - 1
    lower_start = divider + 1
    lower_end = h - 1
    
    if upper_end < upper_start or lower_end < lower_start:
        return [[0]]
    
    # Find column for red (2)
    col_red = -1
    for c in range(w):
        has_red = False
        for r in range(upper_start, upper_end + 1):
            if grid_lst[r][c] == 2:
                has_red = True
                break
        if not has_red:
            for r in range(lower_start, lower_end + 1):
                if grid_lst[r][c] == 2:
                    has_red = True
                    break
        if has_red:
            col_red = c
            break  # assume first one
    
    # Find column for yellow (4)
    col_yellow = -1
    for c in range(w):
        if c == col_red:
            continue
        has_yellow = False
        for r in range(upper_start, upper_end + 1):
            if grid_lst[r][c] == 4:
                has_yellow = True
                break
        if not has_yellow:
            for r in range(lower_start, lower_end + 1):
                if grid_lst[r][c] == 4:
                    has_yellow = True
                    break
        if has_yellow:
            col_yellow = c
            break  # assume one
    
    if col_red == -1 or col_yellow == -1:
        return [[0]]
    
    def get_height(col, C, start_r, end_r):
        height = 0
        for r in range(start_r, end_r + 1):
            if grid_lst[r][col] == C:
                height += 1
            else:
                break
        return height
    
    upper_red_h = get_height(col_red, 2, upper_start, upper_end)
    lower_red_h = get_height(col_red, 2, lower_start, lower_end)
    diff_red = lower_red_h - upper_red_h
    
    upper_yellow_h = get_height(col_yellow, 4, upper_start, upper_end)
    lower_yellow_h = get_height(col_yellow, 4, lower_start, lower_end)
    diff_yellow = lower_yellow_h - upper_yellow_h
    
    if diff_red > diff_yellow:
        C = 2
    elif diff_yellow > diff_red:
        C = 4
    else:
        C = 2  # tie, choose red
    
    return [[C, C], [C, C]]