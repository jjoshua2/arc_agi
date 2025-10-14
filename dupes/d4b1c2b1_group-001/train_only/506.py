import numpy as np
from collections import deque

def transform(grid_lst):
    grid = np.array(grid_lst)
    rows, cols = grid.shape
    
    # Find all purple components using BFS
    visited = np.zeros_like(grid, dtype=bool)
    purple = 8
    red = 2
    
    def bfs(r, c):
        component = []
        queue = deque([(r, c)])
        visited[r, c] = True
        while queue:
            cr, cc = queue.popleft()
            component.append((cr, cc))
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < rows and 0 <= nc < cols and not visited[nr, nc] and grid[nr, nc] == purple:
                    visited[nr, nc] = True
                    queue.append((nr, nc))
        return component
    
    purple_components = []
    for r in range(rows):
        for c in range(cols):
            if grid[r, c] == purple and not visited[r, c]:
                component = bfs(r, c)
                purple_components.append(component)
    
    # For each purple component, find red cells and create symmetric patterns
    for component in purple_components:
        if not component:
            continue
            
        # Get bounding box of the component
        min_r = min(r for r, c in component)
        max_r = max(r for r, c in component)
        min_c = min(c for r, c in component)
        max_c = max(c for r, c in component)
        
        # Find red cells within this component
        red_cells = []
        for r, c in component:
            if grid[r, c] == red:
                red_cells.append((r, c))
        
        # For each red cell, create symmetric patterns
        for r, c in red_cells:
            # Check if this red cell is part of a pattern that needs to be mirrored
            # Look for symmetric positions relative to the center of the component
            
            # Horizontal symmetry
            center_c = (min_c + max_c) // 2
            mirror_c = 2 * center_c - c
            if min_c <= mirror_c <= max_c and grid[r, mirror_c] == 0:
                grid[r, mirror_c] = red
            
            # Vertical symmetry
            center_r = (min_r + max_r) // 2
            mirror_r = 2 * center_r - r
            if min_r <= mirror_r <= max_r and grid[mirror_r, c] == 0:
                grid[mirror_r, c] = red
            
            # Diagonal symmetry (both axes)
            mirror_r2 = 2 * center_r - r
            mirror_c2 = 2 * center_c - c
            if (min_r <= mirror_r2 <= max_r and min_c <= mirror_c2 <= max_c and 
                grid[mirror_r2, mirror_c2] == 0):
                grid[mirror_r2, mirror_c2] = red
    
    return grid.tolist()