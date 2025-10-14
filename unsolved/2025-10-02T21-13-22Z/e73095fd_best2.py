from collections import deque

def transform(grid: list[list[int]]) -> list[list[int]]:
    if not grid or not grid[0]:
        return []
    h = len(grid)
    w = len(grid[0])
    visited = [[False] * w for _ in range(h)]
    queue = deque()
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    
    # Start flood fill from top and bottom 0s
    for c in range(w):
        if grid[0][c] == 0 and not visited[0][c]:
            visited[0][c] = True
            queue.append((0, c))
        if grid[h-1][c] == 0 and not visited[h-1][c]:
            visited[h-1][c] = True
            queue.append((h-1, c))
    
    while queue:
        r, c = queue.popleft()
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not visited[nr][nc] and grid[nr][nc] == 0:
                visited[nr][nc] = True
                queue.append((nr, nc))
    
    # Create output, fill unreached 0s with 4
    output = [row[:] for row in grid]
    for r in range(h):
        for c in range(w):
            if grid[r][c] == 0 and not visited[r][c]:
                output[r][c] = 4
    
    return output