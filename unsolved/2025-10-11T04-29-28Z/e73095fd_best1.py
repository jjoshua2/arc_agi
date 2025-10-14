from collections import deque
from typing import List

def transform(grid: List[List[int]]) -> List[List[int]]:
    if not grid or not grid[0]:
        return grid
    rows = len(grid)
    cols = len(grid[0])
    out = [row[:] for row in grid]

    # Visited mask for zeros reachable from the top row (the "outside" zeros)
    visited = [[False]*cols for _ in range(rows)]
    q = deque()

    # Start BFS from zeros on the top row (row 0) only
    for c in range(cols):
        if grid[0][c] == 0:
            visited[0][c] = True
            q.append((0, c))

    # 4-directional BFS
    while q:
        r, c = q.popleft()
        for dr, dc in ((-1,0),(1,0),(0,-1),(0,1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                if not visited[nr][nc] and grid[nr][nc] == 0:
                    visited[nr][nc] = True
                    q.append((nr, nc))

    # Any zero not visited is an enclosed pocket -> fill with 4
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 0 and not visited[r][c]:
                out[r][c] = 4

    return out