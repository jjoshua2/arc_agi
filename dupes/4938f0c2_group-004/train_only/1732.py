import numpy as np
from collections import deque

def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    if not grid_lst or not grid_lst[0]:
        return []
    
    grid = np.array(grid_lst)
    rows, cols = grid.shape
    visited = np.zeros_like(grid, dtype=bool)
    
    # Directions for 4-way connectivity
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    # Flood fill from borders to mark exterior black cells
    queue = deque()
    
    # Add border cells that are black (0)
    for r in range(rows):
        for c in [0, cols-1]:
            if grid[r][c] == 0 and not visited[r][c]:
                visited[r][c] = True
                queue.append((r, c))
    
    for c in range(cols):
        for r in [0, rows-1]:
            if grid[r][c] == 0 and not visited[r][c]:
                visited[r][c] = True
                queue.append((r, c))
    
    # Flood fill to mark all exterior black cells
    while queue:
        r, c = queue.popleft()
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                if grid[nr][nc] == 0 and not visited[nr][nc]:
                    visited[nr][nc] = True
                    queue.append((nr, nc))
    
    # Create output by filling interior black cells with yellow (4)
    output = grid.copy()
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 0 and not visited[r][c]:
                output[r][c] = 4
    
    return output.tolist()