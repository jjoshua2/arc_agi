import numpy as np
from collections import defaultdict

def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    grid = np.array(grid_lst)
    rows, cols = grid.shape
    
    # Find all border cells (non-zero cells adjacent to 0)
    border_cells = []
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    
    for r in range(rows):
        for c in range(cols):
            if grid[r, c] == 0:
                continue
                
            # Check if adjacent to any 0 cell
            is_border = False
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] == 0:
                    is_border = True
                    break
            
            if is_border:
                border_cells.append((r, c, grid[r, c]))
    
    if not border_cells:
        return grid.tolist()
    
    # Group border cells by color and count
    color_groups = defaultdict(list)
    for r, c, color in border_cells:
        color_groups[color].append((r, c))
    
    # Find color with most border cells
    if not color_groups:
        return grid.tolist()
    
    max_color = max(color_groups.keys(), key=lambda k: len(color_groups[k]))
    max_cells = color_groups[max_color]
    
    # Find bounding box of these cells
    if not max_cells:
        return grid.tolist()
    
    min_r = min(r for r, c in max_cells)
    max_r = max(r for r, c in max_cells)
    min_c = min(c for r, c in max_cells)
    max_c = max(c for r, c in max_cells)
    
    # Create output grid of bounding box size filled with max_color
    output_rows = max_r - min_r + 1
    output_cols = max_c - min_c + 1
    output = [[max_color] * output_cols for _ in range(output_rows)]
    
    return output