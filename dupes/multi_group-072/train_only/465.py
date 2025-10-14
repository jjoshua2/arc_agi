import numpy as np

def find_patterns(grid):
    """Find all 3x3 patterns in the grid"""
    rows, cols = grid.shape
    patterns = []
    
    for r in range(1, rows - 1):
        for c in range(1, cols - 1):
            # Check if this could be a pattern center
            pattern = grid[r-1:r+2, c-1:c+2]
            if np.any(pattern != 0):  # If pattern has non-zero values
                patterns.append((r, c, pattern))
    
    return patterns

def transform(grid_lst):
    grid = np.array(grid_lst)
    rows, cols = grid.shape
    
    # Create output as copy of input
    output = grid.copy()
    
    # Find all existing patterns
    patterns = find_patterns(grid)
    
    if not patterns:
        return grid_lst  # No patterns found, return original
    
    # Use the first pattern found as the template
    _, _, template_pattern = patterns[0]
    
    # Find all colored cells (non-zero)
    colored_cells = []
    for r in range(rows):
        for c in range(cols):
            if grid[r, c] != 0:
                colored_cells.append((r, c, grid[r, c]))
    
    # Stamp the pattern for each colored cell
    for r, c, color in colored_cells:
        # Create colored version of the template pattern
        colored_pattern = np.where(template_pattern != 0, color, 0)
        
        # Stamp the pattern centered at (r, c)
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    if colored_pattern[dr + 1, dc + 1] != 0:
                        output[nr, nc] = colored_pattern[dr + 1, dc + 1]
    
    return output.tolist()