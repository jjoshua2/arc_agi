def transform(grid: list[list[int]]) -> list[list[int]]:
    if not grid or not grid[0]:
        return []
    rows = len(grid)
    cols = len(grid[0])
    
    # Find the non-zero color C
    C = 0
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 0:
                C = grid[r][c]
                break
        if C != 0:
            break
    
    # Create output grid all 0s
    output = [[0 for _ in range(cols)] for _ in range(rows)]
    
    # Paint top and bottom rows
    for c in range(cols):
        output[0][c] = C
        output[rows - 1][c] = C
    
    # Paint left and right columns
    for r in range(rows):
        output[r][0] = C
        output[r][cols - 1] = C
    
    return output