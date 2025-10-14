import numpy as np
from collections import deque

def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    if not grid_lst or not grid_lst[0]:
        return grid_lst
    
    grid = np.array(grid_lst, dtype=int)
    rows, cols = grid.shape
    
    # Directions for 4-connected adjacency
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    # BFS to mark exterior 0's (reachable from border)
    visited = np.zeros((rows, cols), dtype=bool)
    q = deque()
    
    # Enqueue all border 0's
    for r in range(rows):
        for c in [0, cols - 1]:
            if grid[r, c] == 0 and not visited[r, c]:
                visited[r, c] = True
                q.append((r, c))
    for c in range(cols):
        for r in [0, rows - 1]:
            if grid[r, c] == 0 and not visited[r, c]:
                visited[r, c] = True
                q.append((r, c))
    
    # Flood fill from border 0's
    while q:
        r, c = q.popleft()
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] == 0 and not visited[nr, nc]:
                visited[nr, nc] = True
                q.append((nr, nc))
    
    # Set unvisited 0's (enclosed) to 3
    for r in range(rows):
        for c in range(cols):
            if grid[r, c] == 0 and not visited[r, c]:
                grid[r, c] = 3
    
    return grid.tolist()