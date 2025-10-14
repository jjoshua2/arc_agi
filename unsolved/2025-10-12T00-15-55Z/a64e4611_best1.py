import numpy as np
from collections import deque

def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    if not grid_lst or not grid_lst[0]:
        return []
    grid = np.array(grid_lst, dtype=int)
    rows, cols = grid.shape
    visited = np.zeros((rows, cols), dtype=bool)
    q = deque()
    # Add all border 0s to queue
    for i in range(rows):
        if grid[i, 0] == 0 and not visited[i, 0]:
            q.append((i, 0))
            visited[i, 0] = True
        if grid[i, cols - 1] == 0 and not visited[i, cols - 1]:
            q.append((i, cols - 1))
            visited[i, cols - 1] = True
    for j in range(1, cols - 1):  # Avoid corners duplicate
        if grid[0, j] == 0 and not visited[0, j]:
            q.append((0, j))
            visited[0, j] = True
        if grid[rows - 1, j] == 0 and not visited[rows - 1, j]:
            q.append((rows - 1, j))
            visited[rows - 1, j] = True
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    while q:
        x, y = q.popleft()
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < rows and 0 <= ny < cols and not visited[nx, ny] and grid[nx, ny] == 0:
                visited[nx, ny] = True
                q.append((nx, ny))
    # Fill unmarked 0s (holes) with 3
    for i in range(rows):
        for j in range(cols):
            if grid[i, j] == 0 and not visited[i, j]:
                grid[i, j] = 3
    return grid.tolist()