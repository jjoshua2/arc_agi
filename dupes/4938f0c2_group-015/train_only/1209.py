from collections import deque
from typing import List

def transform(grid: List[List[int]]) -> List[List[int]]:
    if not grid or not grid[0]:
        return []
    rows = len(grid)
    cols = len(grid[0])
    visited = [[False] * cols for _ in range(rows)]
    q = deque()
    # Add border 5's to queue
    for i in range(rows):
        if grid[i][0] == 5 and not visited[i][0]:
            visited[i][0] = True
            q.append((i, 0))
        if grid[i][cols - 1] == 5 and not visited[i][cols - 1]:
            visited[i][cols - 1] = True
            q.append((i, cols - 1))
    for j in range(cols):
        if grid[0][j] == 5 and not visited[0][j]:
            visited[0][j] = True
            q.append((0, j))
        if grid[rows - 1][j] == 5 and not visited[rows - 1][j]:
            visited[rows - 1][j] = True
            q.append((rows - 1, j))
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    while q:
        r, c = q.popleft()
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc] and grid[nr][nc] == 5:
                visited[nr][nc] = True
                q.append((nr, nc))
    # Create output by changing unreached 5's to 8
    output = [row[:] for row in grid]
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == 5 and not visited[i][j]:
                output[i][j] = 8
    return output