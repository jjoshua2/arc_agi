import numpy as np
from collections import deque

def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    grid = np.array(grid_lst)
    rows, cols = grid.shape
    
    # Create visited matrix
    visited = np.zeros_like(grid, dtype=bool)
    
    # Directions for 4-connectivity
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    # Find all connected components of same non-zero color
    for i in range(rows):
        for j in range(cols):
            if grid[i, j] != 0 and not visited[i, j]:
                color = grid[i, j]
                # BFS to find connected component of same color
                component = []
                queue = deque([(i, j)])
                visited[i, j] = True
                
                while queue:
                    r, c = queue.popleft()
                    component.append((r, c))
                    
                    for dr, dc in directions:
                        nr, nc = r + dr, c + dc
                        if (0 <= nr < rows and 0 <= nc < cols and 
                            not visited[nr, nc] and grid[nr, nc] == color):
                            visited[nr, nc] = True
                            queue.append((nr, nc))
                
                # If component size < 3, change to 3
                if len(component) < 3:
                    for r, c in component:
                        grid[r, c] = 3
    
    return grid.tolist()