import numpy as np

def transform(grid_lst):
    grid = np.array(grid_lst)
    rows, cols = grid.shape
    
    # Find purple (8) cells
    purple_mask = grid == 8
    purple_rows, purple_cols = np.where(purple_mask)
    
    if len(purple_rows) == 0:
        return grid.tolist()
    
    # Find bounding box of purple area
    min_purple_row, max_purple_row = np.min(purple_rows), np.max(purple_rows)
    min_purple_col, max_purple_col = np.min(purple_cols), np.max(purple_cols)
    
    # Create output grid
    output = grid.copy()
    
    # Process each colored shape (non-zero, non-purple)
    for color in range(1, 10):
        if color == 8:
            continue
            
        color_mask = grid == color
        color_rows, color_cols = np.where(color_mask)
        
        if len(color_rows) == 0:
            continue
            
        # For each colored cell, determine its position relative to purple area
        for r, c in zip(color_rows, color_cols):
            # Check if this colored cell is near the purple area
            if (r < min_purple_row or r > max_purple_row or 
                c < min_purple_col or c > max_purple_col):
                continue
                
            # Determine fill pattern based on position
            if r < min_purple_row:  # Above purple
                fill_color = 3  # Green
                fill_rows = range(r, min_purple_row)
                fill_cols = range(min_purple_col, max_purple_col + 1)
            elif r > max_purple_row:  # Below purple
                fill_color = 4  # Yellow
                fill_rows = range(max_purple_row + 1, r + 1)
                fill_cols = range(min_purple_col, max_purple_col + 1)
            elif c < min_purple_col:  # Left of purple
                fill_color = 3  # Green
                fill_rows = range(min_purple_row, max_purple_row + 1)
                fill_cols = range(c, min_purple_col)
            elif c > max_purple_col:  # Right of purple
                fill_color = 4  # Yellow
                fill_rows = range(min_purple_row, max_purple_row + 1)
                fill_cols = range(max_purple_col + 1, c + 1)
            else:
                continue
                
            # Fill the area
            for fr in fill_rows:
                for fc in fill_cols:
                    if 0 <= fr < rows and 0 <= fc < cols and output[fr, fc] == 0:
                        output[fr, fc] = fill_color
    
    return output.tolist()