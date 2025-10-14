import collections

def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    if not grid_lst:
        return []
    grid = [row[:] for row in grid_lst]
    rows = len(grid)
    if rows == 0:
        return grid
    cols = len(grid[0])
    visited = [[False] * cols for _ in range(rows)]
    q = collections.deque()
    # Enqueue 0s on top and bottom rows
    for j in range(cols):
        if grid[0][j] == 0 and not visited[0][j]:
            visited[0][j] = True
            q.append((0, j))
        if rows > 1 and grid[rows-1][j] == 0 and not visited[rows-1][j]:
            visited[rows-1][j] = True
            q.append((rows-1, j))
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    while q:
        r, c = q.popleft()
        for dr, dc in directions:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc] and grid[nr][nc] == 0:
                visited[nr][nc] = True
                q.append((nr, nc))
    # Fill unvisited 0s with 4
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == 0 and not visited[i][j]:
                grid[i][j] = 4
    return grid