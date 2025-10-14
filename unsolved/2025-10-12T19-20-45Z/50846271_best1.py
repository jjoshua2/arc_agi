def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    if not grid_lst or not grid_lst[0]:
        return []
    rows = len(grid_lst)
    cols = len(grid_lst[0])
    out = [row[:] for row in grid_lst]
    
    # Horizontal gap filling
    for r in range(rows):
        two_cols = [c for c in range(cols) if grid_lst[r][c] == 2]
        if len(two_cols) < 2:
            continue
        two_cols = sorted(set(two_cols))
        for i in range(len(two_cols) - 1):
            for c in range(two_cols[i] + 1, two_cols[i + 1]):
                if out[r][c] == 5:
                    out[r][c] = 8
    
    # Vertical gap filling
    for c in range(cols):
        two_rows_list = [r for r in range(rows) if grid_lst[r][c] == 2]
        if len(two_rows_list) < 2:
            continue
        two_rows_list = sorted(set(two_rows_list))
        for i in range(len(two_rows_list) - 1):
            for rr in range(two_rows_list[i] + 1, two_rows_list[i + 1]):
                if out[rr][c] == 5:
                    out[rr][c] = 8
    
    # Vertical group extensions
    for c in range(cols):
        r = 0
        while r < rows:
            if grid_lst[r][c] != 2:
                r += 1
                continue
            # Start of a vertical group of 2's
            min_r = r
            r += 1
            while r < rows and grid_lst[r][c] == 2:
                r += 1
            max_r = r - 1
            height = max_r - min_r + 1
            
            # Top extension only if height == 1 and right is 5 (in out)
            if height == 1:
                if c + 1 < cols and out[min_r][c + 1] == 5:
                    tr = min_r - 1
                    if tr >= 0 and out[tr][c] == 5:
                        out[tr][c] = 8
            
            # Bottom extension if left is 5 (in out)
            if c - 1 >= 0 and out[max_r][c - 1] == 5:
                br = max_r + 1
                if br < rows and out[br][c] == 5:
                    out[br][c] = 8
    
    return out