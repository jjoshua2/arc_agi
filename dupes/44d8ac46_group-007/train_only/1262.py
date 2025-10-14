from collections import deque

def transform(grid: list[list[int]]) -> list[list[int]]:
    if not grid or not grid[0]:
        return []
    rows = len(grid)
    cols = len(grid[0])
    output = [row[:] for row in grid]
    visited = [[False] * cols for _ in range(rows)]
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # Queue for BFS
    q = deque()

    # Start from all border 0 cells
    for r in range(rows):
        for c in [0, cols - 1]:  # Left and right borders
            if grid[r][c] == 0 and not visited[r][c]:
                visited[r][c] = True
                q.append((r, c))
        if r in [0, rows - 1]:  # Top and bottom borders, but skip corners already added
            for c in range(1, cols - 1):
                if grid[r][c] == 0 and not visited[r][c]:
                    visited[r][c] = True
                    q.append((r, c))

    # BFS to mark all reachable 0 cells from border
    while q:
        r, c = q.popleft()
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 0 and not visited[nr][nc]:
                visited[nr][nc] = True
                q.append((nr, nc))

    # Fill unvisited 0 cells with 2
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 0 and not visited[r][c]:
                output[r][c] = 2

    return output