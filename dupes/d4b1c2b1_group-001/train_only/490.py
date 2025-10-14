import numpy as np
from collections import defaultdict

def transform(grid_lst):
    if not grid_lst or not grid_lst[0]:
        return grid_lst
    
    grid = np.array(grid_lst)
    rows, cols = grid.shape
    
    # Create output as copy of input
    output = grid.copy()
    
    # Find all colored cells (non-0, non-1)
    colored_cells = defaultdict(list)
    for r in range(rows):
        for c in range(cols):
            color = grid[r, c]
            if color != 0 and color != 1:
                colored_cells[color].append((r, c))
    
    # For each color, process its pattern
    for color, cells in colored_cells.items():
        if len(cells) < 2:
            continue
            
        # Find bounding box of this color
        min_r = min(r for r, c in cells)
        max_r = max(r for r, c in cells)
        min_c = min(c for r, c in cells)
        max_c = max(c for r, c in cells)
        
        # Create a pattern based on the relative positions
        pattern = set()
        for r, c in cells:
            # Calculate relative position within bounding box
            rel_r = r - min_r
            rel_c = c - min_c
            
            # Add this relative position to pattern
            pattern.add((rel_r, rel_c))
        
        # Tile the pattern across the bounding box
        pattern_height = max_r - min_r + 1
        pattern_width = max_c - min_c + 1
        
        for r in range(min_r, max_r + 1):
            for c in range(min_c, max_c + 1):
                rel_r = r - min_r
                rel_c = c - min_c
                
                # Check if this relative position is in the pattern
                if (rel_r % pattern_height, rel_c % pattern_width) in pattern:
                    output[r, c] = color
    
    return output.tolist()