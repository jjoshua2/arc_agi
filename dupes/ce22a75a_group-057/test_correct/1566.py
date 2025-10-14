def transform(grid: list[list[int]]) -> list[list[int]]:
    rows = len(grid)
    cols = len(grid[0])
    
    # Create output grid with same dimensions as input, filled with 0s
    output = [[0] * cols for _ in range(rows)]
    
    # For each grey cell (color 5), place a 3x3 blue block (color 1) centered at that position
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == 5:
                # Place 3x3 block centered at (i,j)
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < rows and 0 <= nj < cols:
                            output[ni][nj] = 1
    
    return output