def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    if not grid_lst or len(grid_lst) != 6 or len(grid_lst[0]) != 5:
        return grid_lst  # Assuming always 6x5, but safeguard
    
    top = grid_lst[:3]
    bottom = grid_lst[3:]
    
    output = [[0 for _ in range(5)] for _ in range(3)]
    
    for i in range(3):
        for j in range(5):
            has_top = top[i][j] != 0
            has_bottom = bottom[i][j] != 0
            if has_top != has_bottom:
                output[i][j] = 6
    
    return output