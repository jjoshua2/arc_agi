def transform(grid: list[list[int]]) -> list[list[int]]:
    if not grid or len(grid) < 13 or len(grid[0]) != 4:
        raise ValueError("Invalid grid size")
    
    upper = grid[:6]
    lower = grid[7:13]
    
    output = [[0 for _ in range(4)] for _ in range(6)]
    
    for i in range(6):
        for j in range(4):
            if upper[i][j] == 0 and lower[i][j] == 0:
                output[i][j] = 8
            # else remains 0
    
    return output