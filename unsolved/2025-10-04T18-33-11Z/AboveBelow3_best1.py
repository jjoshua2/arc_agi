import numpy as np

def transform(grid_lst):
    if not grid_lst or not grid_lst[0]:
        return []
    
    grid = np.array(grid_lst)
    rows, cols = grid.shape
    
    # Create a copy to work with
    result = grid.copy()
    
    # Simulate gravity - make blocks fall down
    for c in range(cols):
        # For each column, process from bottom to top
        for r in range(rows-2, -1, -1):  # from second last row to top
            if result[r, c] != 0:  # if there's a block
                current_r = r
                # Try to move it down as far as possible
                while current_r + 1 < rows and result[current_r + 1, c] == 0:
                    # Move block down
                    result[current_r + 1, c] = result[current_r, c]
                    result[current_r, c] = 0
                    current_r += 1
    
    # Remove empty rows from top and bottom
    # Find first non-empty row
    first_non_empty = 0
    while first_non_empty < rows and np.all(result[first_non_empty, :] == 0):
        first_non_empty += 1
    
    # Find last non-empty row
    last_non_empty = rows - 1
    while last_non_empty >= 0 and np.all(result[last_non_empty, :] == 0):
        last_non_empty -= 1
    
    # If all rows are empty, return empty grid
    if first_non_empty > last_non_empty:
        return []
    
    # Extract the non-empty rows
    result = result[first_non_empty:last_non_empty+1, :]
    
    return result.tolist()