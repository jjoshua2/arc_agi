def transform(grid: list[list[int]]) -> list[list[int]]:
    if not grid or not grid[0]:
        return []
    h = len(grid)
    w = len(grid[0])
    output = [[0] * w for _ in range(h)]
    
    # Fill qualifying rows
    for r in range(h):
        left = grid[r][0]
        right = grid[r][w - 1]
        if left == right and left > 0:
            c = left
            for j in range(w):
                output[r][j] = c
    
    # Fill qualifying columns
    for k in range(w):
        top = grid[0][k]
        bottom = grid[h - 1][k]
        if top == bottom and top > 0:
            c = top
            for i in range(h):
                output[i][k] = c
    
    return output