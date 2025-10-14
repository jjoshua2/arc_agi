from collections import deque
from typing import List

def transform(grid: List[List[int]]) -> List[List[int]]:
    if not grid or not grid[0]:
        return []
    h = len(grid)
    w = len(grid[0])
    output = [row[:] for row in grid]
    visited = [[False] * w for _ in range(h)]
    
    # Directions for BFS
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    # Flood fill from borders to mark exterior 0's
    q = deque()
    # Top and bottom rows
    for j in range(w):
        if grid[0][j] == 0 and not visited[0][j]:
            q.append((0, j))
            visited[0][j] = True
        if grid[h-1][j] == 0 and not visited[h-1][j]:
            q.append((h-1, j))
            visited[h-1][j] = True
    # Left and right columns (avoid duplicating corners)
    for i in range(1, h-1):
        if grid[i][0] == 0 and not visited[i][0]:
            q.append((i, 0))
            visited[i][0] = True
        if grid[i][w-1] == 0 and not visited[i][w-1]:
            q.append((i, w-1))
            visited[i][w-1] = True
    
    while q:
        r, c = q.popleft()
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and grid[nr][nc] == 0 and not visited[nr][nc]:
                visited[nr][nc] = True
                q.append((nr, nc))
    
    # Now fill interior 0's (unvisited 0's) with 3
    for i in range(h):
        for j in range(w):
            if grid[i][j] == 0 and not visited[i][j]:
                # Flood fill this interior component
                interior_q = deque([(i, j)])
                visited[i][j] = True
                output[i][j] = 3
                while interior_q:
                    r, c = interior_q.popleft()
                    for dr, dc in directions:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < h and 0 <= nc < w and grid[nr][nc] == 0 and not visited[nr][nc]:
                            visited[nr][nc] = True
                            output[nr][nc] = 3
                            interior_q.append((nr, nc))
    
    return output