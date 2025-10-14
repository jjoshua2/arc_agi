import numpy as np

def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    grid = np.array(grid_lst)
    rows, cols = grid.shape
    
    # Create output grid (copy of input)
    output = grid.copy()
    
    # Find all 2x2 grey blocks (color 5)
    for r in range(rows - 1):
        for c in range(cols - 1):
            # Check if this is a 2x2 grey block
            if (grid[r, c] == 5 and grid[r, c+1] == 5 and 
                grid[r+1, c] == 5 and grid[r+1, c+1] == 5):
                
                # Add colored cells around the block
                # Top-left (color 1)
                if r-1 >= 0 and c-1 >= 0 and output[r-1, c-1] == 0:
                    output[r-1, c-1] = 1
                
                # Top-right (color 2)
                if r-1 >= 0 and c+2 < cols and output[r-1, c+2] == 0:
                    output[r-1, c+2] = 2
                
                # Bottom-left (color 3)
                if r+2 < rows and c-1 >= 0 and output[r+2, c-1] == 0:
                    output[r+2, c-1] = 3
                
                # Bottom-right (color 4)
                if r+2 < rows and c+2 < cols and output[r+2, c+2] == 0:
                    output[r+2, c+2] = 4
    
    return output.tolist()