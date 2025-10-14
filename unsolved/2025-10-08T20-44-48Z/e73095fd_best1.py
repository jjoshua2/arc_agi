import numpy as np
from collections import deque

def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    grid = np.array(grid_lst)
    rows, cols = grid.shape
    visited = np.zeros((rows, cols), dtype=bool)
    
    # Directions for 4-connected movement
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    # Queue for BFS
    queue = deque()
    
    # Enqueue all border 0 cells and mark them visited
    for r in range(rows):
        # Top and bottom rows
        if grid[r, 0] == 0:
            queue.append((r, 0))
            visited[r, 0] = True
        if grid[r, cols - 1] == 0:
            queue.append((r, cols - 1))
            visited[r, cols - 1] = True
    
    for c in range(1, cols - 1):  # Avoid duplicating corners
        # Left and right columns
        if grid[0, c] == 0:
            queue.append((0, c))
            visited[0, c] = True
        if grid[rows - 1, c] == 0:
            queue.append((rows - 1, c))
            visited[rows - 1, c] = True
    
    # BFS to mark all reachable 0 cells
    while queue:
        cr, cc = queue.popleft()
        for dr, dc in directions:
            nr, nc = cr + dr, cc + dc
            if (0 <= nr < rows and 0 <= nc < cols and
                grid[nr, nc] == 0 and not visited[nr, nc]):
                visited[nr, nc] = True
                queue.append((nr, nc))
    
    # Set unvisited 0 cells (enclosed) to 4
    for r in range(rows):
        for c in range(cols):
            if grid[r, c] == 0 and not visited[r, c]:
                grid[r, c] = 4
    
    return grid.tolist()