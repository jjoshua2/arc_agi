from collections import deque

def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    if not grid_lst or not grid_lst[0]:
        return []
    h = len(grid_lst)
    w = len(grid_lst[0])
    grid = [row[:] for row in grid_lst]
    
    visited = [[False] * w for _ in range(h)]
    queue = deque()
    
    # Add all border 5s to queue
    for i in range(h):
        if grid[i][0] == 5:
            queue.append((i, 0))
            visited[i][0] = True
        if grid[i][w - 1] == 5:
            queue.append((i, w - 1))
            visited[i][w - 1] = True
    for j in range(w):
        if grid[0][j] == 5:
            queue.append((0, j))
            visited[0][j] = True
        if grid[h - 1][j] == 5:
            queue.append((h - 1, j))
            visited[h - 1][j] = True
    
    # BFS flood fill through 5s
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    while queue:
        r, c = queue.popleft()
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and grid[nr][nc] == 5 and not visited[nr][nc]:
                visited[nr][nc] = True
                queue.append((nr, nc))
    
    # Change unvisited 5s (enclosed) to 8
    for i in range(h):
        for j in range(w):
            if grid[i][j] == 5 and not visited[i][j]:
                grid[i][j] = 8
    
    return grid