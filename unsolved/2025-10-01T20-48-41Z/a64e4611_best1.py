from collections import deque
from typing import List

def transform(grid_lst: List[List[int]]) -> List[List[int]]:
    if not grid_lst or not grid_lst[0]:
        return []
    
    rows = len(grid_lst)
    cols = len(grid_lst[0])
    
    visited = [[False] * cols for _ in range(rows)]
    queue = deque()
    
    # Directions for 4-connectivity
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    # Flood fill from border zeros
    for i in range(rows):
        # Top row
        if grid_lst[0][i] == 0:
            queue.append((0, i))
            visited[0][i] = True
        # Bottom row
        if grid_lst[rows - 1][i] == 0:
            queue.append((rows - 1, i))
            visited[rows - 1][i] = True
    
    for i in range(cols):
        # Left column (skip corners already added)
        if grid_lst[i][0] == 0 and not visited[i][0]:
            queue.append((i, 0))
            visited[i][0] = True
        # Right column (skip corners already added)
        if grid_lst[i][cols - 1] == 0 and not visited[i][cols - 1]:
            queue.append((i, cols - 1))
            visited[i][cols - 1] = True
    
    # BFS to mark all reachable zeros from the borders
    while queue:
        x, y = queue.popleft()
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if (0 <= nx < rows and 0 <= ny < cols and
                grid_lst[nx][ny] == 0 and not visited[nx][ny]):
                visited[nx][ny] = True
                queue.append((nx, ny))
    
    # Create output by filling unvisited zeros with 3
    output = [row[:] for row in grid_lst]
    for r in range(rows):
        for c in range(cols):
            if grid_lst[r][c] == 0 and not visited[r][c]:
                output[r][c] = 3
    
    return output