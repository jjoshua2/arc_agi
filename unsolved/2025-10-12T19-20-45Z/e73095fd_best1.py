import numpy as np
from collections import deque

def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    grid = np.array(grid_lst)
    rows, cols = grid.shape
    
    visited = np.zeros((rows, cols), dtype=bool)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    components = []  # list of (size, list of (r,c))
    
    for i in range(rows):
        for j in range(cols):
            if grid[i, j] == 0 and not visited[i, j]:
                # BFS to find component
                component_size = 0
                component_cells = []
                queue = deque([(i, j)])
                visited[i, j] = True
                
                while queue:
                    r, c = queue.popleft()
                    component_size += 1
                    component_cells.append((r, c))
                    
                    for dr, dc in directions:
                        nr, nc = r + dr, c + dc
                        if (0 <= nr < rows and 0 <= nc < cols and 
                            not visited[nr, nc] and grid[nr, nc] == 0):
                            visited[nr, nc] = True
                            queue.append((nr, nc))
                
                components.append((component_size, component_cells))
    
    if not components:
        return grid.tolist()
    
    # Find max size
    max_size = max(size for size, _ in components)
    
    # Set all non-max components to 4
    for size, cells in components:
        if size != max_size:
            for r, c in cells:
                grid[r, c] = 4
    
    return grid.tolist()