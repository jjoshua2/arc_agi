def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    if not grid_lst:
        return []
    rows = len(grid_lst)
    if rows == 0:
        return []
    cols = len(grid_lst[0])
    grid = [row[:] for row in grid_lst]
    output = [row[:] for row in grid]
    
    to_fill = set()
    
    max_k = min(rows, cols)
    for k in range(1, max_k + 1):
        for r in range(rows - k + 1):
            for c in range(cols - k + 1):
                # Check all cells in block are 0
                all_zero = True
                for i in range(k):
                    for j in range(k):
                        if grid[r + i][c + j] != 0:
                            all_zero = False
                            break
                    if not all_zero:
                        break
                if not all_zero:
                    continue
                
                # Check perimeters
                valid = True
                
                # Top
                if r > 0:
                    for j in range(k):
                        if grid[r - 1][c + j] != 5:
                            valid = False
                            break
                else:
                    valid = False
                if not valid:
                    continue
                
                # Bottom
                if r + k < rows:
                    for j in range(k):
                        if grid[r + k][c + j] != 5:
                            valid = False
                            break
                else:
                    valid = False
                if not valid:
                    continue
                
                # Left
                if c > 0:
                    for i in range(k):
                        if grid[r + i][c - 1] != 5:
                            valid = False
                            break
                else:
                    valid = False
                if not valid:
                    continue
                
                # Right
                if c + k < cols:
                    for i in range(k):
                        if grid[r + i][c + k] != 5:
                            valid = False
                            break
                else:
                    valid = False
                if not valid:
                    continue
                
                # If valid, add all cells in the block to to_fill
                if valid:
                    for i in range(k):
                        for j in range(k):
                            to_fill.add((r + i, c + j))
    
    # Fill the cells
    for pos in to_fill:
        i, j = pos
        output[i][j] = 2
    
    return output