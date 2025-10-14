import numpy as np
from collections import deque

def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    if not grid_lst or not grid_lst[0]:
        return []
    grid = np.array(grid_lst, dtype=int)
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    q = deque()
    # Add all border 0's to queue
    for i in range(h):
        for j in [0, w-1]:
            if grid[i, j] == 0 and not visited[i, j]:
                q.append((i, j))
                visited[i, j] = True
    for j in range(w):
        if grid[0, j] == 0 and not visited[0, j]:
            q.append((0, j))
            visited[0, j] = True
        if grid[h-1, j] == 0 and not visited[h-1, j]:
            q.append((h-1, j))
            visited[h-1, j] = True
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    while q:
        x, y = q.popleft()
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < h and 0 <= ny < w and not visited[nx, ny] and grid[nx, ny] == 0:
                visited[nx, ny] = True
                q.append((nx, ny))
    # Now fill unvisited 0's with 8
    for i in range(h):
        for j in range(w):
            if grid[i, j] == 0 and not visited[i, j]:
                grid[i, j] = 8
    return grid.tolist()