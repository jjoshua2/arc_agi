import numpy as np
from collections import deque

def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    grid = np.array(grid_lst)
    rows, cols = grid.shape
    output = np.zeros_like(grid)
    
    # Find all connected components of non-zero cells
    visited = np.zeros_like(grid, dtype=bool)
    components = []
    
    for r in range(rows):
        for c in range(cols):
            if grid[r, c] != 0 and not visited[r, c]:
                # BFS to find connected component
                component = []
                queue = deque([(r, c)])
                visited[r, c] = True
                
                while queue:
                    cr, cc = queue.popleft()
                    component.append((cr, cc))
                    
                    for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                        nr, nc = cr + dr, cc + dc
                        if (0 <= nr < rows and 0 <= nc < cols and 
                            grid[nr, nc] != 0 and not visited[nr, nc]):
                            visited[nr, nc] = True
                            queue.append((nr, nc))
                
                components.append(component)
    
    # Calculate center of the grid
    center_r = rows // 2
    center_c = cols // 2
    
    # Copy original patterns to output
    for component in components:
        for r, c in component:
            output[r, c] = grid[r, c]
    
    # Create mirror images of each component
    for component in components:
        # Find bounding box of component
        min_r = min(r for r, c in component)
        max_r = max(r for r, c in component)
        min_c = min(c for r, c in component)
        max_c = max(c for r, c in component)
        
        # Calculate center of component
        comp_center_r = (min_r + max_r) // 2
        comp_center_c = (min_c + max_c) // 2
        
        # Calculate offset from grid center
        offset_r = comp_center_r - center_r
        offset_c = comp_center_c - center_c
        
        # Mirror position
        mirror_r = center_r - offset_r
        mirror_c = center_c - offset_c
        
        # Calculate translation vector
        trans_r = mirror_r - comp_center_r
        trans_c = mirror_c - comp_center_c
        
        # Copy component to mirror position
        for r, c in component:
            nr = r + trans_r
            nc = c + trans_c
            if 0 <= nr < rows and 0 <= nc < cols:
                output[nr, nc] = grid[r, c]
    
    return output.tolist()