import numpy as np
from collections import deque

def transform(grid_lst):
    if not grid_lst or not grid_lst[0]:
        return grid_lst
    
    grid = np.array(grid_lst)
    h, w = grid.shape
    output = grid.copy()
    purple = 8
    
    # Find all connected components of purple cells
    visited = np.zeros_like(grid, dtype=bool)
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    
    for i in range(h):
        for j in range(w):
            if grid[i, j] == purple and not visited[i, j]:
                # BFS to find the component
                component = []
                queue = deque([(i, j)])
                visited[i, j] = True
                
                while queue:
                    x, y = queue.popleft()
                    component.append((x, y))
                    
                    for dx, dy in directions:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < h and 0 <= ny < w and not visited[nx, ny] and grid[nx, ny] == purple:
                            visited[nx, ny] = True
                            queue.append((nx, ny))
                
                # Find boundary cells of this component
                boundary_cells = set()
                for x, y in component:
                    for dx, dy in directions:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < h and 0 <= ny < w and grid[nx, ny] != purple:
                            boundary_cells.add((x, y))
                            break
                
                # Recolor boundary cells based on row position
                for x, y in boundary_cells:
                    color = (x * 4 // h) + 1
                    output[x, y] = color
    
    return output.tolist()