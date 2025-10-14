from collections import deque
import copy

def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    if not grid_lst or not grid_lst[0]:
        return []
    rows = len(grid_lst)
    cols = len(grid_lst[0])
    grid = copy.deepcopy(grid_lst)
    visited = [[False] * cols for _ in range(rows)]
    q = deque()
    # Enqueue all border 0's
    for i in range(rows):
        if grid[i][0] == 0:
            q.append((i, 0))
            visited[i][0] = True
        if grid[i][cols - 1] == 0:
            q.append((i, cols - 1))
            visited[i][cols - 1] = True
    for j in range(1, cols - 1):  # Avoid duplicates
        if grid[0][j] == 0:
            q.append((0, j))
            visited[0][j] = True
        if grid[rows - 1][j] == 0:
            q.append((rows - 1, j))
            visited[rows - 1][j] = True
    # 4-connected directions
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    while q:
        r, c = q.popleft()
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc] and grid[nr][nc] == 0:
                visited[nr][nc] = True
                q.append((nr, nc))
    # Fill unmarked 0's with 3
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == 0 and not visited[i][j]:
                grid[i][j] = 3
    return grid