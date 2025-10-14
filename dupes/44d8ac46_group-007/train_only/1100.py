from collections import deque
from typing import List

def transform(grid: List[List[int]]) -> List[List[int]]:
    if not grid or not grid[0]:
        return [row[:] for row in grid]
    
    rows = len(grid)
    cols = len(grid[0])
    
    # Create a copy to avoid modifying the original during flood fill
    output = [row[:] for row in grid]
    
    # Directions for 4-way connectivity
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    # Visited matrix
    visited = [[False] * cols for _ in range(rows)]
    
    # Queue for BFS flood fill from borders
    q = deque()
    
    # Enqueue all border 0's
    for r in range(rows):
        # Top and bottom borders
        if output[r][0] == 0:
            q.append((r, 0))
            visited[r][0] = True
        if output[r][cols - 1] == 0:
            q.append((r, cols - 1))
            visited[r][cols - 1] = True
    
    for c in range(cols):
        # Left and right borders (avoid duplicating corners)
        if output[0][c] == 0 and not visited[0][c]:
            q.append((0, c))
            visited[0][c] = True
        if output[rows - 1][c] == 0 and not visited[rows - 1][c]:
            q.append((rows - 1, c))
            visited[rows - 1][c] = True
    
    # Flood fill: mark all reachable 0's
    while q:
        x, y = q.popleft()
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if (0 <= nx < rows and 0 <= ny < cols and
                not visited[nx][ny] and output[nx][ny] == 0):
                visited[nx][ny] = True
                q.append((nx, ny))
    
    # Now fill any unvisited 0's with 2 (enclosed regions)
    for r in range(rows):
        for c in range(cols):
            if output[r][c] == 0 and not visited[r][c]:
                output[r][c] = 2
    
    return output