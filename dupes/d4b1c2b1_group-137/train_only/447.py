import numpy as np
from collections import deque

def transform(grid_lst):
    if not grid_lst or not grid_lst[0]:
        return grid_lst
    
    grid = np.array(grid_lst)
    h, w = grid.shape
    
    # Find all blue cells (value 1)
    blue_cells = np.argwhere(grid == 1)
    
    # If no blue cells, return original grid
    if len(blue_cells) == 0:
        return grid.tolist()
    
    # Create a copy to modify
    result = grid.copy()
    
    # Directions for 4-connected neighbors
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    
    # Perform BFS from each blue cell
    visited = np.zeros_like(grid, dtype=bool)
    queue = deque()
    
    # Add all blue cells to the queue
    for r, c in blue_cells:
        queue.append((r, c))
        visited[r, c] = True
    
    while queue:
        r, c = queue.popleft()
        
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            
            # Check if within bounds
            if 0 <= nr < h and 0 <= nc < w:
                # If empty cell and not visited, fill it and add to queue
                if result[nr, nc] == 0 and not visited[nr, nc]:
                    result[nr, nc] = 1
                    visited[nr, nc] = True
                    queue.append((nr, nc))
                # If obstacle (5 or 8), don't cross it
                # (already handled by the condition above)
    
    return result.tolist()