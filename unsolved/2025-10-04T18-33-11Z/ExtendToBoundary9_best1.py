import numpy as np

def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    grid = np.array(grid_lst)
    rows, cols = grid.shape
    
    # Find the frame color (non-zero, non-black color that forms a rectangle)
    frame_color = None
    for i in range(rows):
        for j in range(cols):
            if grid[i, j] != 0:
                frame_color = grid[i, j]
                break
        if frame_color is not None:
            break
    
    if frame_color is None:
        return grid.tolist()
    
    # Find frame boundaries
    frame_cells = np.argwhere(grid == frame_color)
    if len(frame_cells) == 0:
        return grid.tolist()
    
    min_row, min_col = np.min(frame_cells, axis=0)
    max_row, max_col = np.max(frame_cells, axis=0)
    
    # Find interior shape color (non-zero, non-frame color inside frame)
    interior_color = None
    for i in range(min_row + 1, max_row):
        for j in range(min_col + 1, max_col):
            if grid[i, j] != 0 and grid[i, j] != frame_color:
                interior_color = grid[i, j]
                break
        if interior_color is not None:
            break
    
    if interior_color is None:
        return grid.tolist()
    
    # Find interior shape bounding box
    interior_cells = np.argwhere((grid == interior_color) & 
                                (np.arange(rows)[:, None] >= min_row + 1) & 
                                (np.arange(rows)[:, None] <= max_row - 1) & 
                                (np.arange(cols) >= min_col + 1) & 
                                (np.arange(cols) <= max_col - 1))
    
    if len(interior_cells) == 0:
        return grid.tolist()
    
    interior_min_row, interior_min_col = np.min(interior_cells, axis=0)
    interior_max_row, interior_max_col = np.max(interior_cells, axis=0)
    
    # Create output grid
    output = grid.copy()
    
    # Fill cells within frame that are in same row or column as interior shape
    for i in range(min_row + 1, max_row):
        for j in range(min_col + 1, max_col):
            if (i >= interior_min_row and i <= interior_max_row) or (j >= interior_min_col and j <= interior_max_col):
                if output[i, j] == 0:  # Only fill black cells
                    output[i, j] = interior_color
    
    return output.tolist()