from collections import deque

def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    grid = [row[:] for row in grid_lst]
    h = len(grid)
    if h == 0:
        return []
    w = len(grid[0])
    if w == 0:
        return [[] for _ in range(h)]

    visited = [[False] * w for _ in range(h)]
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    q = deque()

    # Enqueue all border 0 cells
    for r in range(h):
        for c in (0, w - 1):
            if grid[r][c] == 0:
                visited[r][c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if grid[r][c] == 0:
                visited[r][c] = True
                q.append((r, c))

    # BFS to mark reachable 0 cells
    while q:
        r, c = q.popleft()
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and grid[nr][nc] == 0 and not visited[nr][nc]:
                visited[nr][nc] = True
                q.append((nr, nc))

    # Set unvisited 0 to 3
    for r in range(h):
        for c in range(w):
            if grid[r][c] == 0 and not visited[r][c]:
                grid[r][c] = 3

    return grid