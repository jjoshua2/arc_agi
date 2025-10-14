import numpy as np
from collections import deque

def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    if not grid_lst or not grid_lst[0]:
        return []
    grid = np.array(grid_lst)
    rows, cols = grid.shape
    
    # Directions for 8-way neighbors and 4-way flood
    dirs8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    dirs4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    visited = np.zeros((rows, cols), dtype=bool)
    
    # Queue for BFS
    queue = deque()
    
    # Find starting points: border 0 cells with at least one non-zero (colored) 8-neighbor
    for i in range(rows):
        for j in range(cols):
            if grid[i, j] != 0:
                continue
            is_border = (i == 0 or i == rows - 1 or j == 0 or j == cols - 1)
            if not is_border:
                continue
            has_colored = False
            for di, dj in dirs8:
                ni, nj = i + di, j + dj
                if 0 <= ni < rows and 0 <= nj < cols and grid[ni, nj] != 0:
                    has_colored = True
                    break
            if has_colored:
                queue.append((i, j))
                visited[i, j] = True
    
    # Flood fill using 4-connectivity
    while queue:
        r, c = queue.popleft()
        for dr, dc in dirs4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] == 0 and not visited[nr, nc]:
                visited[nr, nc] = True
                queue.append((nr, nc))
    
    # Change all unmarked 0 to 3
    for i in range(rows):
        for j in range(cols):
            if grid[i, j] == 0 and not visited[i, j]:
                grid[i, j] = 3
    
    return grid.tolist()