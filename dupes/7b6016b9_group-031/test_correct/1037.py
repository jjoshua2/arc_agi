from collections import deque
from typing import List

def transform(grid_lst: List[List[int]]) -> List[List[int]]:
    if not grid_lst or not grid_lst[0]:
        return []
    rows = len(grid_lst)
    cols = len(grid_lst[0])
    grid = [row[:] for row in grid_lst]
    visited = [[False] * cols for _ in range(rows)]
    q = deque()

    # Enqueue all border 0s and set to 3
    for i in range(rows):
        if grid[i][0] == 0 and not visited[i][0]:
            q.append((i, 0))
            visited[i][0] = True
            grid[i][0] = 3
        if grid[i][cols - 1] == 0 and not visited[i][cols - 1]:
            q.append((i, cols - 1))
            visited[i][cols - 1] = True
            grid[i][cols - 1] = 3
    for j in range(cols):
        if grid[0][j] == 0 and not visited[0][j]:
            q.append((0, j))
            visited[0][j] = True
            grid[0][j] = 3
        if grid[rows - 1][j] == 0 and not visited[rows - 1][j]:
            q.append((rows - 1, j))
            visited[rows - 1][j] = True
            grid[rows - 1][j] = 3

    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 4-connected

    while q:
        r, c = q.popleft()
        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc] and grid[nr][nc] == 0:
                visited[nr][nc] = True
                grid[nr][nc] = 3
                q.append((nr, nc))

    # Fill remaining 0s with 2
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == 0:
                grid[i][j] = 2

    return grid