def transform(grid: list[list[int]]) -> list[list[int]]:
    if not grid or not grid[0]:
        return []
    rows = len(grid)
    cols = len(grid[0])
    output = [[0] * cols for _ in range(rows)]
    
    # Fill horizontal lines
    for i in range(rows):
        left = grid[i][0]
        right = grid[i][cols - 1]
        if left == right and left != 0:
            C = left
            for j in range(cols):
                output[i][j] = C
    
    # Fill vertical lines
    for j in range(cols):
        top = grid[0][j]
        bottom = grid[rows - 1][j]
        if top == bottom and top != 0:
            C = top
            for i in range(rows):
                output[i][j] = C
    
    return output