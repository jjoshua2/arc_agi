from collections import deque

def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    if not grid_lst or not grid_lst[0]:
        return []
    grid = [row[:] for row in grid_lst]
    rows = len(grid)
    cols = len(grid[0])
    if rows == 0 or cols == 0:
        return grid
    
    visited = [[False] * cols for _ in range(rows)]
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    q = deque()
    
    # Add all border 0 cells to queue and mark visited
    for i in range(rows):
        if grid[i][0] == 0 and not visited[i][0]:
            visited[i][0] = True
            q.append((i, 0))
        if grid[i][cols - 1] == 0 and not visited[i][cols - 1]:
            visited[i][cols - 1] = True
            q.append((i, cols - 1))
    for j in range(cols):
        if grid[0][j] == 0 and not visited[0][j]:
            visited[0][j] = True
            q.append((0, j))
        if grid[rows - 1][j] == 0 and not visited[rows - 1][j]:
            visited[rows - 1][j] = True
            q.append((rows - 1, j))
    
    # BFS to mark all reachable 0 cells
    while q:
        r, c = q.popleft()
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 0 and not visited[nr][nc]:
                visited[nr][nc] = True
                q.append((nr, nc))
    
    # Change unmarked 0 cells to 2
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == 0 and not visited[i][j]:
                grid[i][j] = 2
    
    return grid