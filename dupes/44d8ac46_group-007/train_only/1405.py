import collections

def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    if not grid_lst or not grid_lst[0]:
        return grid_lst
    h = len(grid_lst)
    w = len(grid_lst[0])
    grid = [row[:] for row in grid_lst]
    visited = [[False] * w for _ in range(h)]
    q = collections.deque()

    # Enqueue all border 0s
    # Top row
    for j in range(w):
        if grid[0][j] == 0:
            visited[0][j] = True
            q.append((0, j))
    # Bottom row
    for j in range(w):
        if grid[h-1][j] == 0:
            visited[h-1][j] = True
            q.append((h-1, j))
    # Left column, excluding corners
    for i in range(1, h-1):
        if grid[i][0] == 0:
            visited[i][0] = True
            q.append((i, 0))
    # Right column, excluding corners
    for i in range(1, h-1):
        if grid[i][w-1] == 0:
            visited[i][w-1] = True
            q.append((i, w-1))

    # Directions for 4-way connectivity
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # BFS to mark all reachable 0s from border
    while q:
        r, c = q.popleft()
        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not visited[nr][nc] and grid[nr][nc] == 0:
                visited[nr][nc] = True
                q.append((nr, nc))

    # Fill unvisited 0s (enclosed regions) with 2
    for i in range(h):
        for j in range(w):
            if grid[i][j] == 0 and not visited[i][j]:
                grid[i][j] = 2

    return grid