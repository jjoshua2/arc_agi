import numpy as np
from collections import deque

def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    grid = np.array(grid_lst)
    rows, cols = grid.shape
    
    # Find connected components of value 8
    visited = np.zeros_like(grid, dtype=bool)
    components = []
    
    # Directions for 4-connectivity
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 8 and not visited[r][c]:
                # Found a new component
                component = []
                queue = deque([(r, c)])
                visited[r][c] = True
                
                while queue:
                    cr, cc = queue.popleft()
                    component.append((cr, cc))
                    
                    for dr, dc in directions:
                        nr, nc = cr + dr, cc + dc
                        if (0 <= nr < rows and 0 <= nc < cols and 
                            grid[nr][nc] == 8 and not visited[nr][nc]):
                            visited[nr][nc] = True
                            queue.append((nr, nc))
                
                components.append(component)
    
    n = len(components)
    
    # Create n x n diagonal matrix with 8s on diagonal
    result = np.zeros((n, n), dtype=int)
    np.fill_diagonal(result, 8)
    
    return result.tolist()