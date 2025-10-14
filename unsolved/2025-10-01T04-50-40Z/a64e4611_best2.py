from collections import deque
from typing import List

def transform(grid: List[List[int]]) -> List[List[int]]:
    if not grid or not grid[0]:
        return []
    rows, cols = len(grid), len(grid[0])
    output = [row[:] for row in grid]
    visited = [[False] * cols for _ in range(rows)]
    
    q = deque()
    
    # Add all border 0 cells to the queue
    for i in range(rows):
        if grid[i][0] == 0:
            q.append((i, 0))
            visited[i][0] = True
        if grid[i][cols - 1] == 0:
            q.append((i, cols - 1))
            visited[i][cols - 1] = True
    
    for j in range(cols):
        if grid[0][j] == 0:
            q.append((0, j))
            visited[0][j] = True
        if grid[rows - 1][j] == 0:
            q.append((rows - 1, j))
            visited[rows - 1][j] = True
    
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    # BFS to mark all reachable 0 cells from borders
    while q:
        r, c = q.popleft()
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 0 and not visited[nr][nc]:
                visited[nr][nc] = True
                q.append((nr, nc))
    
    # Fill unreachable 0 cells with 3
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 0 and not visited[r][c]:
                output[r][c] = 3
    
    return output