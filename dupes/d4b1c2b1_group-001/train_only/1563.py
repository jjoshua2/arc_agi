import numpy as np

def transform(grid_lst):
    grid = np.array(grid_lst)
    h, w = grid.shape
    
    # Create output grid
    output = grid.copy()
    
    # Find all red cells (value 2)
    red_cells = np.argwhere(grid == 2)
    
    for r, c in red_cells:
        # Check if this red cell is part of a horizontal block
        if c > 0 and grid[r, c-1] > 0 and c < w-1 and grid[r, c+1] > 0:
            # Extend horizontally to the left
            left_extend = c - 1
            while left_extend >= 0 and output[r, left_extend] == 0:
                output[r, left_extend] = grid[r, c+1]  # Use color from right side
                left_extend -= 1
            
            # Extend horizontally to the right
            right_extend = c + 1
            while right_extend < w and output[r, right_extend] == 0:
                output[r, right_extend] = grid[r, c-1]  # Use color from left side
                right_extend += 1
        
        # Check if this red cell is part of a vertical block
        elif r > 0 and grid[r-1, c] > 0 and r < h-1 and grid[r+1, c] > 0:
            # Extend vertically upward
            up_extend = r - 1
            while up_extend >= 0 and output[up_extend, c] == 0:
                output[up_extend, c] = grid[r+1, c]  # Use color from below
                up_extend -= 1
            
            # Extend vertically downward
            down_extend = r + 1
            while down_extend < h and output[down_extend, c] == 0:
                output[down_extend, c] = grid[r-1, c]  # Use color from above
                down_extend += 1
    
    return output.tolist()