def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    if not grid_lst or not grid_lst[0]:
        return []
    h = len(grid_lst)
    w = len(grid_lst[0])
    output = [row[:] for row in grid_lst]
    
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if (grid_lst[r][c-1] == 0 and
                grid_lst[r][c] == 0 and
                grid_lst[r][c+1] == 0 and
                grid_lst[r-1][c] == 0 and
                grid_lst[r+1][c] == 0):
                output[r-1][c] = 3
                output[r][c-1] = 3
                output[r][c] = 3
                output[r][c+1] = 3
                output[r+1][c] = 3
    
    return output