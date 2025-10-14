import numpy as np

def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    grid = np.array(grid_lst)
    rows, cols = grid.shape
    
    # Create a copy of the grid to modify
    result = grid.copy()
    
    # Find all 2x2 blocks of grey cells (color 5)
    for r in range(rows - 1):
        for c in range(cols - 1):
            # Check if this is a 2x2 block of grey cells
            if (grid[r, c] == 5 and grid[r, c+1] == 5 and 
                grid[r+1, c] == 5 and grid[r+1, c+1] == 5):
                
                # Place colored cells diagonally around the block
                # Top-left corner (color 1)
                if r > 0 and c > 0 and result[r-1, c-1] == 0:
                    result[r-1, c-1] = 1
                
                # Top-right corner (color 2)
                if r > 0 and c+2 < cols and result[r-1, c+2] == 0:
                    result[r-1, c+2] = 2
                
                # Bottom-left corner (color 3)
                if r+2 < rows and c > 0 and result[r+2, c-1] == 0:
                    result[r+2, c-1] = 3
                
                # Bottom-right corner (color 4)
                if r+2 < rows and c+2 < cols and result[r+2, c+2] == 0:
                    result[r+2, c+2] = 4
    
    return result.tolist()