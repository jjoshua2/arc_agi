from collections import deque
from typing import List

def transform(grid: List[List[int]]) -> List[List[int]]:
    if not grid or not grid[0]:
        return []
    h = len(grid)
    w = len(grid[0])
    output = [row[:] for row in grid]
    visited = [[False] * w for _ in range(h)]
    q = deque()
    # Add all border 0's to queue
    for r in range(h):
        if grid[r][0] == 0 and not visited[r][0]:
            q.append((r, 0))
            visited[r][0] = True
        if grid[r][w-1] == 0 and not visited[r][w-1]:
            q.append((r, w-1))
            visited[r][w-1] = True
    for c in range(w):
        if grid[0][c] == 0 and not visited[0][c]:
            q.append((0, c))
            visited[0][c] = True
        if grid[h-1][c] == 0 and not visited[h-1][c]:
            q.append((h-1, c))
            visited[h-1][c] = True
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    while q:
        r, c = q.popleft()
        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not visited[nr][nc] and grid[nr][nc] == 0:
                visited[nr][nc] = True
                q.append((nr, nc))
    # Set unvisited 0's to 8
    for r in range(h):
        for c in range(w):
            if grid[r][c] == 0 and not visited[r][c]:
                output[r][c] = 8
    return output