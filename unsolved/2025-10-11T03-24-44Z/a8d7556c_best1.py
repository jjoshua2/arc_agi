from typing import List
import collections

def transform(grid_lst: List[List[int]]) -> List[List[int]]:
    if not grid_lst or not grid_lst[0]:
        return grid_lst
    
    rows = len(grid_lst)
    cols = len(grid_lst[0])
    grid = [row[:] for row in grid_lst]
    visited = [[False] * cols for _ in range(rows)]
    queue = collections.deque()
    
    # Enqueue all border 0 cells
    for r in range(rows):
        for c in (0, cols - 1):
            if grid[r][c] == 0 and not visited[r][c]:
                visited[r][c] = True
                queue.append((r, c))
    for c in range(cols):
        for r in (0, rows - 1):
            if grid[r][c] == 0 and not visited[r][c]:
                visited[r][c] = True
                queue.append((r, c))
    
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    while queue:
        r, c = queue.popleft()
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 0 and not visited[nr][nc]:
                visited[nr][nc] = True
                queue.append((nr, nc))
    
    # Fill unvisited 0s (enclosed) with 2
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 0 and not visited[r][c]:
                grid[r][c] = 2
    
    return grid