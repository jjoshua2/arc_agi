import numpy as np

def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    grid = np.array(grid_lst)
    output = grid.copy()
    rows, cols = grid.shape
    
    # Find the bounding box of the main colored region (non-zero cells)
    non_zero_mask = grid != 0
    if not np.any(non_zero_mask):
        return grid.tolist()
    
    non_zero_coords = np.argwhere(non_zero_mask)
    min_row, min_col = np.min(non_zero_coords, axis=0)
    max_row, max_col = np.max(non_zero_coords, axis=0)
    
    # Identify isolated cells (not in the main bounding box)
    isolated_cells = []
    for r in range(rows):
        for c in range(cols):
            if grid[r, c] != 0 and (r < min_row or r > max_row or c < min_col or c > max_col):
                isolated_cells.append((r, c, grid[r, c]))
    
    # Create border around main region
    border_cells = []
    
    # Top border
    if min_row > 0:
        for c in range(min_col, max_col + 1):
            border_cells.append((min_row - 1, c))
    
    # Bottom border
    if max_row < rows - 1:
        for c in range(min_col, max_col + 1):
            border_cells.append((max_row + 1, c))
    
    # Left border
    if min_col > 0:
        for r in range(min_row, max_row + 1):
            border_cells.append((r, min_col - 1))
    
    # Right border
    if max_col < cols - 1:
        for r in range(min_row, max_row + 1):
            border_cells.append((r, max_col + 1))
    
    # Corners
    if min_row > 0 and min_col > 0:
        border_cells.append((min_row - 1, min_col - 1))
    if min_row > 0 and max_col < cols - 1:
        border_cells.append((min_row - 1, max_col + 1))
    if max_row < rows - 1 and min_col > 0:
        border_cells.append((max_row + 1, min_col - 1))
    if max_row < rows - 1 and max_col < cols - 1:
        border_cells.append((max_row + 1, max_col + 1))
    
    # For each border cell, find the nearest isolated cell and use its color
    for br, bc in border_cells:
        min_dist = float('inf')
        best_color = 0
        
        for ir, ic, color in isolated_cells:
            dist = abs(br - ir) + abs(bc - ic)
            if dist < min_dist:
                min_dist = dist
                best_color = color
        
        if best_color != 0:
            output[br, bc] = best_color
    
    return output.tolist()