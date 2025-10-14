import numpy as np
from collections import deque

def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    if not grid_lst:
        return []
    grid = np.array(grid_lst)
    rows, cols = grid.shape
    if rows == 0 or cols == 0:
        return []
    
    visited = np.zeros((rows, cols), dtype=bool)
    
    def bfs(start_r, start_c):
        queue = deque([(start_r, start_c)])
        visited[start_r, start_c] = True
        while queue:
            x, y = queue.popleft()
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < rows and 0 <= ny < cols and not visited[nx, ny] and grid[nx, ny] == 8:
                    visited[nx, ny] = True
                    queue.append((nx, ny))
    
    n_components = 0
    for i in range(rows):
        for j in range(cols):
            if grid[i, j] == 8 and not visited[i, j]:
                bfs(i, j)
                n_components += 1
    
    if n_components == 0:
        return []
    
    output = [[8 if row == col else 0 for col in range(n_components)] for row in range(n_components)]
    return output