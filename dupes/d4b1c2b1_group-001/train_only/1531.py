import numpy as np

def transform(grid_list):
    grid = np.array(grid_list)
    # Find the central region: largest connected component of value 8
    # First, get all cells with value 8
    mask = (grid == 8)
    # Label connected components
    from scipy.ndimage import label
    labeled, num_features = label(mask)
    # Find the largest component
    if num_features == 0:
        return grid.tolist()  # no change
    sizes = np.bincount(labeled.ravel())[1:]
    if len(sizes) == 0:
        return grid.tolist()
    largest_label = np.argmax(sizes) + 1
    central_region = (labeled == largest_label)
    # Get bounding box of central region
    rows, cols = grid.shape
    coords = np.argwhere(central_region)
    min_row, min_col = np.min(coords, axis=0)
    max_row, max_col = np.max(coords, axis=0)
    
    output_grid = grid.copy()
    # For each cell not in {0,8} that is adjacent to the central region
    for r in range(rows):
        for c in range(cols):
            if grid[r, c] not in {0, 8} and not central_region[r, c]:
                # Check if it is adjacent to the central region (Manhattan distance 1)
                if (r < min_row or r > max_row or c < min_col or c > max_col):
                    continue  # not adjacent
                # Check which side it is on
                if r < min_row:  # above
                    target_r = min_row
                    target_c = c
                elif r > max_row:  # below
                    target_r = max_row
                    target_c = c
                elif c < min_col:  # left
                    target_r = r
                    target_c = min_col
                elif c > max_col:  # right
                    target_r = r
                    target_c = max_col
                else:
                    continue  # inside, not adjacent
                # Now, in the output, we want to move the color to the target cell
                # So, set the target cell to the color, and set the original cell to 0
                output_grid[target_r, target_c] = grid[r, c]
                output_grid[r, c] = 0
    return output_grid.tolist()