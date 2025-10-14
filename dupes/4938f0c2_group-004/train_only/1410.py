from collections import deque

def transform(grid: list[list[int]]) -> list[list[int]]:
    if not grid or not grid[0]:
        return []
    rows = len(grid)
    cols = len(grid[0])
    output = [row[:] for row in grid]
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    visited = [[False] * cols for _ in range(rows)]
    q = deque()
    # Add border 0s to queue
    for i in range(rows):
        if output[i][0] == 0:
            q.append((i, 0))
            visited[i][0] = True
        if output[i][cols - 1] == 0:
            q.append((i, cols - 1))
            visited[i][cols - 1] = True
    for j in range(1, cols - 1):
        if output[0][j] == 0:
            q.append((0, j))
            visited[0][j] = True
        if output[rows - 1][j] == 0:
            q.append((rows - 1, j))
            visited[rows - 1][j] = True
    # BFS to mark outside 0s
    while q:
        r, c = q.popleft()
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc] and output[nr][nc] == 0:
                visited[nr][nc] = True
                q.append((nr, nc))
    # Fill unvisited 0s with 4
    for i in range(rows):
        for j in range(cols):
            if output[i][j] == 0 and not visited[i][j]:
                output[i][j] = 4
    return output