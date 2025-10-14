from collections import deque
from typing import List
import copy

def transform(grid_lst: List[List[int]]) -> List[List[int]]:
    if not grid_lst or not grid_lst[0]:
        return []
    grid = copy.deepcopy(grid_lst)
    rows = len(grid)
    cols = len(grid[0])
    visited = [[False] * cols for _ in range(rows)]
    q = deque()
    
    # Enqueue all border 0's
    for i in range(rows):
        if grid[i][0] == 0 and not visited[i][0]:
            q.append((i, 0))
            visited[i][0] = True
        if grid[i][cols - 1] == 0 and not visited[i][cols - 1]:
            q.append((i, cols - 1))
            visited[i][cols - 1] = True
    for j in range(cols):
        if grid[0][j] == 0 and not visited[0][j]:
            q.append((0, j))
            visited[0][j] = True
        if grid[rows - 1][j] == 0 and not visited[rows - 1][j]:
            q.append((rows - 1, j))
            visited[rows - 1][j] = True
    
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    while q:
        r, c = q.popleft()
        # Do not change grid[r][c] here; just visit
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 0 and not visited[nr][nc]:
                visited[nr][nc] = True
                q.append((nr, nc))
    
    # Set unvisited 0's (enclosed) to 2
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == 0 and not visited[i][j]:
                grid[i][j] = 2
    
    return grid