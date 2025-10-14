from collections import deque
from typing import List

def transform(grid: List[List[int]]) -> List[List[int]]:
    if not grid or not grid[0]:
        return []
    
    rows = len(grid)
    cols = len(grid[0])
    output = [row[:] for row in grid]
    visited = [[False for _ in range(cols)] for _ in range(rows)]
    
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    
    # Flood fill from all border 0 cells
    queue = deque()
    
    # Top and bottom rows
    for c in range(cols):
        if grid[0][c] == 0:
            queue.append((0, c))
            visited[0][c] = True
        if grid[rows - 1][c] == 0:
            queue.append((rows - 1, c))
            visited[rows - 1][c] = True
    
    # Left and right columns (excluding corners already added)
    for r in range(1, rows - 1):
        if grid[r][0] == 0:
            queue.append((r, 0))
            visited[r][0] = True
        if grid[r][cols - 1] == 0:
            queue.append((r, cols - 1))
            visited[r][cols - 1] = True
    
    while queue:
        r, c = queue.popleft()
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if (0 <= nr < rows and 0 <= nc < cols and
                not visited[nr][nc] and grid[nr][nc] == 0):
                visited[nr][nc] = True
                queue.append((nr, nc))
    
    # Change unvisited 0s to 4
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 0 and not visited[r][c]:
                output[r][c] = 4
    
    return output