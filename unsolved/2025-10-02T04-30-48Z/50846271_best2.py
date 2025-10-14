from collections import deque
import copy

def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    if not grid_lst or not grid_lst[0]:
        return []
    
    rows = len(grid_lst)
    cols = len(grid_lst[0])
    grid = copy.deepcopy(grid_lst)  # Work on a copy to avoid modifying input
    visited = [[False for _ in range(cols)] for _ in range(rows)]
    
    # Directions for adjacent cells
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    # Queue for BFS
    q = deque()
    
    # Enqueue all border cells that are 0 or 5 (traversable, not 2)
    for r in range(rows):
        # Left border
        if grid[r][0] != 2 and not visited[r][0]:
            q.append((r, 0))
            visited[r][0] = True
        # Right border
        if grid[r][cols-1] != 2 and not visited[r][cols-1]:
            q.append((r, cols-1))
            visited[r][cols-1] = True
    
    for c in range(1, cols-1):  # Avoid corners duplicate
        # Top border
        if grid[0][c] != 2 and not visited[0][c]:
            q.append((0, c))
            visited[0][c] = True
        # Bottom border
        if grid[rows-1][c] != 2 and not visited[rows-1][c]:
            q.append((rows-1, c))
            visited[rows-1][c] = True
    
    # Flood fill: traverse through 0 and 5 cells
    while q:
        r, c = q.popleft()
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if (0 <= nr < rows and 0 <= nc < cols and
                not visited[nr][nc] and
                grid[nr][nc] != 2 and  # Can't pass through 2
                grid[nr][nc] in [0, 5]):  # Only traverse 0 and 5
                visited[nr][nc] = True
                q.append((nr, nc))
    
    # Now, change unvisited 5 cells to 8
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 5 and not visited[r][c]:
                grid[r][c] = 8
    
    return grid