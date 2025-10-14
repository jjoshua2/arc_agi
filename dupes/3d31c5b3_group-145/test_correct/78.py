def transform(grid: list[list[int]]) -> list[list[int]]:
    if not grid or len(grid) != 12 or len(grid[0]) != 6:
        raise ValueError("Input must be 12x6 grid")
    
    output = [[0 for _ in range(6)] for _ in range(3)]
    
    for k in range(3):
        for j in range(6):
            a = grid[k][j]
            if a != 0:
                output[k][j] = a
                continue
            
            b = grid[3 + k][j]
            if b != 0:
                output[k][j] = b
                continue
            
            c = grid[6 + k][j]
            if c != 0:
                d = grid[9 + k][j]
                if d != 0:
                    output[k][j] = d
                else:
                    output[k][j] = c
                continue
            
            d = grid[9 + k][j]
            if d != 0:
                output[k][j] = d
            # else 0, already set
    
    return output