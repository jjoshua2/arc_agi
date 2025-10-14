import numpy as np

def transform(grid_lst):
    grid = np.array(grid_lst)
    h, w = grid.shape
    
    # Create output grid
    output = grid.copy()
    
    # Find all purple cells (8)
    purple_positions = np.argwhere(grid == 8)
    
    # Find all red cells (2)
    red_positions = np.argwhere(grid == 2)
    
    if len(purple_positions) == 0:
        return output.tolist()
    
    # Sort red positions by row, then by column
    red_positions = red_positions[np.lexsort((red_positions[:, 1], red_positions[:, 0]))]
    
    # For each purple cell, create a wave pattern
    for purple_row, purple_col in purple_positions:
        # Find the nearest red cell in the same column (for vertical reflection)
        same_col_reds = red_positions[red_positions[:, 1] == purple_col]
        if len(same_col_reds) > 0:
            distances = np.abs(same_col_reds[:, 0] - purple_row)
            nearest_red_row = same_col_reds[np.argmin(distances), 0]
            
            # Create vertical reflection pattern
            for r in range(h):
                if r != purple_row and output[r, purple_col] == 0:
                    # Calculate reflection distance
                    reflection_dist = abs(r - purple_row)
                    if reflection_dist <= abs(nearest_red_row - purple_row):
                        output[r, purple_col] = 8
        
        # Find the nearest red cell in the same row (for horizontal reflection)
        same_row_reds = red_positions[red_positions[:, 0] == purple_row]
        if len(same_row_reds) > 0:
            distances = np.abs(same_row_reds[:, 1] - purple_col)
            nearest_red_col = same_row_reds[np.argmin(distances), 1]
            
            # Create horizontal reflection pattern
            for c in range(w):
                if c != purple_col and output[purple_row, c] == 0:
                    # Calculate reflection distance
                    reflection_dist = abs(c - purple_col)
                    if reflection_dist <= abs(nearest_red_col - purple_col):
                        output[purple_row, c] = 8
    
    return output.tolist()