from collections import deque

def transform(grid: list[list[int]]) -> list[list[int]]:
    if not grid or not grid[0]:
        return grid
    rows = len(grid)
    cols = len(grid[0])
    output = [row[:] for row in grid]
    visited = [[False] * cols for _ in range(rows)]
    queue = deque()
    # Add border 0s
    for i in range(rows):
        if grid[i][0] == 0:
            queue.append((i, 0))
            visited[i][0] = True
        if grid[i][cols - 1] == 0:
            queue.append((i, cols - 1))
            visited[i][cols - 1] = True
    for j in range(1, cols - 1):  # avoid corners duplicate
        if grid[0][j] == 0:
            queue.append((0, j))
            visited[0][j] = True
        if grid[rows - 1][j] == 0:
            queue.append((rows - 1, j))
            visited[rows - 1][j] = True
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    while queue:
        r, c = queue.popleft()
        output[r][c] = 3
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc] and grid[nr][nc] == 0:
                visited[nr][nc] = True
                queue.append((nr, nc))
    # Set remaining 0s to 2
    for i in range(rows):
        for j in range(cols):
            if output[i][j] == 0:
                output[i][j] = 2
    return output