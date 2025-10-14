import numpy as np

def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    grid = np.array(grid_lst)
    
    # Find all non-zero cells
    non_zero_mask = grid != 0
    if not np.any(non_zero_mask):
        return grid.tolist()
    
    # Find the bounding box of non-zero region
    rows, cols = np.where(non_zero_mask)
    min_row, max_row = np.min(rows), np.max(rows)
    min_col, max_col = np.min(cols), np.max(cols)
    
    # Get edge cells (top, bottom, left, right boundaries of the non-zero region)
    edge_cells = []
    
    # Top and bottom edges
    for col in range(min_col, max_col + 1):
        if min_row >= 0 and min_row < grid.shape[0]:
            edge_cells.append(grid[min_row, col])
        if max_row >= 0 and max_row < grid.shape[0]:
            edge_cells.append(grid[max_row, col])
    
    # Left and right edges
    for row in range(min_row, max_row + 1):
        if min_col >= 0 and min_col < grid.shape[1]:
            edge_cells.append(grid[row, min_col])
        if max_col >= 0 and max_col < grid.shape[1]:
            edge_cells.append(grid[row, max_col])
    
    # Remove zeros from edge cells
    edge_cells = [cell for cell in edge_cells if cell != 0]
    
    if not edge_cells:
        return grid.tolist()
    
    # Find the most common color on the edges
    dominant_color = max(set(edge_cells), key=edge_cells.count)
    
    # Create output by replacing all non-zero cells with the dominant color
    output = grid.copy()
    output[non_zero_mask] = dominant_color
    
    return output.tolist()