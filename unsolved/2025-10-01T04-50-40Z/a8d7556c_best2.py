import numpy as np
from collections import deque

def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    grid = np.array(grid_lst, dtype=int)
    rows, cols = grid.shape
    if rows == 0 or cols == 0:
        return []
    
    visited = np.zeros((rows, cols), dtype=bool)
    q = deque()
    
    # Enqueue all border 0's and mark them visited
    for i in range(rows):
        if grid[i, 0] == 0 and not visited[i, 0]:
            q.append((i, 0))
            visited[i, 0] = True
        if grid[i, cols - 1] == 0 and not visited[i, cols - 1]:
            q.append((i, cols - 1))
            visited[i, cols - 1] = True
    for j in range(cols):
        if grid[0, j] == 0 and not visited[0, j]:
            q.append((0, j))
            visited[0, j] = True
        if grid[rows - 1, j] == 0 and not visited[rows - 1, j]:
            q.append((rows - 1, j))
            visited[rows - 1, j] = True
    
    # Directions for 4-connectivity
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    # BFS to mark all external 0's
    while q:
        r, c = q.popleft()
        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and not visited[nr, nc] and grid[nr, nc] == 0:
                visited[nr, nc] = True
                q.append((nr, nc))
    
    # Fill unmarked 0's (enclosed) with 2
    for i in range(rows):
        for j in range(cols):
            if grid[i, j] == 0 and not visited[i, j]:
                grid[i, j] = 2
    
    return grid.tolist()