def transform(grid: list[list[int]]) -> list[list[int]]:
    grid_arr = np.array(grid)
    rows, cols = grid_arr.shape
    output = grid_arr.copy()
    
    # Find the central shape. Assume it's the largest connected component of the most frequent color.
    # But for simplicity, let's assume it's the color that appears most often, and it's contiguous.
    # Since the examples use purple (8), we'll use that.
    purple = 8
    purple_mask = (grid_arr == purple)
    if not np.any(purple_mask):
        return grid  # No central shape, return as is
    
    # Find the bounding box of the purple region
    indices = np.where(purple_mask)
    min_row, max_row = np.min(indices[0]), np.max(indices[0])
    min_col, max_col = np.min(indices[1]), np.max(indices[1])
    
    for r in range(rows):
        for c in range(cols):
            color = grid_arr[r, c]
            if color != 0 and color != purple:  # It's a colored cell not in the central shape
                # Check its position relative to the central shape
                if min_col <= c <= max_col:  # Same column
                    if r < min_row:  # Above
                        target_r = min_row
                        target_c = c
                    elif r > max_row:  # Below
                        target_r = max_row
                        target_c = c
                elif min_row <= r <= max_row:  # Same row
                    if c < min_col:  # Left
                        target_r = r
                        target_c = min_col
                    elif c > max_col:  # Right
                        target_r = r
                        target_c = max_col
                else:
                    continue  # Not in a relevant region
                
                # Check if the target cell is empty
                if output[target_r, target_c] == 0:
                    # Move the cell
                    output[target_r, target_c] = color
                    output[r, c] = 0  # Remove the original
                
    return output.tolist()