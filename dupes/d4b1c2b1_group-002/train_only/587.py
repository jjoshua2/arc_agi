from collections import deque
from typing import List

def transform(grid: List[List[int]]) -> List[List[int]]:
    if not grid or not grid[0]:
        return []
    
    rows = len(grid)
    cols = len(grid[0])
    output = [row[:] for row in grid]
    
    # Directions for 4-connectivity
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    
    # Visited set or matrix for 0 cells
    visited = [[False] * cols for _ in range(rows)]
    
    # Queue for BFS
    queue = deque()
    
    # Add all border 0 cells to queue
    # Top and bottom rows
    for j in range(cols):
        if grid[0][j] == 0 and not visited[0][j]:
            queue.append((0, j))
            visited[0][j] = True
        if grid[rows - 1][j] == 0 and not visited[rows - 1][j]:
            queue.append((rows - 1, j))
            visited[rows - 1][j] = True
    
    # Left and right columns (excluding corners already added)
    for i in range(1, rows - 1):
        if grid[i][0] == 0 and not visited[i][0]:
            queue.append((i, 0))
            visited[i][0] = True
        if grid[i][cols - 1] == 0 and not visited[i][cols - 1]:
            queue.append((i, cols - 1))
            visited[i][cols - 1] = True
    
    # Flood fill through 0 cells only
    while queue:
        x, y = queue.popleft()
        # Mark in output? No, we'll mark visited; change later
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if (0 <= nx < rows and 0 <= ny < cols and
                grid[nx][ny] == 0 and not visited[nx][ny]):
                visited[nx][ny] = True
                queue.append((nx, ny))
    
    # Now, change all unvisited 0 cells to 8
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == 0 and not visited[i][j]:
                output[i][j] = 8
    
    return output