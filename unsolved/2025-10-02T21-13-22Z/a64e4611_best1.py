import numpy as np
from collections import deque

def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    grid = np.array(grid_lst)
    rows, cols = grid.shape
    output = grid.copy()
    
    # Directions for 4-connected movement
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    
    # Visited array for zeros only (use -1 to mark visited zeros)
    visited = np.full((rows, cols), False)
    
    # Queue for BFS
    queue = deque()
    
    # Add all border zeros to queue and mark visited
    for i in range(rows):
        # Top and bottom rows
        if grid[i, 0] == 0:
            queue.append((i, 0))
            visited[i, 0] = True
        if grid[i, cols - 1] == 0:
            queue.append((i, cols - 1))
            visited[i, cols - 1] = True
    
    for j in range(1, cols - 1):  # Avoid corners as already added
        # Left and right columns
        if grid[0, j] == 0:
            queue.append((0, j))
            visited[0, j] = True
        if grid[rows - 1, j] == 0:
            queue.append((rows - 1, j))
            visited[rows - 1, j] = True
    
    # BFS flood fill through zeros (non-zeros are walls)
    while queue:
        x, y = queue.popleft()
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if (0 <= nx < rows and 0 <= ny < cols and
                grid[nx, ny] == 0 and not visited[nx, ny]):
                visited[nx, ny] = True
                queue.append((nx, ny))
    
    # Fill unvisited zeros (enclosed) with 3
    for i in range(rows):
        for j in range(cols):
            if grid[i, j] == 0 and not visited[i, j]:
                output[i, j] = 3
    
    return output.tolist()