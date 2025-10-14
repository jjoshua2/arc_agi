from collections import deque
import copy

def transform(grid: list[list[int]]) -> list[list[int]]:
    if not grid or not grid[0]:
        return []
    rows = len(grid)
    cols = len(grid[0])
    output = copy.deepcopy(grid)
    
    # Directions for 4-connectivity
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    # Visited matrix
    visited = [[False for _ in range(cols)] for _ in range(rows)]
    
    # Queue for BFS
    queue = deque()
    
    # Four corners
    corners = [(0, 0), (0, cols - 1), (rows - 1, 0), (rows - 1, cols - 1)]
    
    # Start from corners if they are 0
    for r, c in corners:
        if 0 <= r < rows and 0 <= c < cols and output[r][c] == 0 and not visited[r][c]:
            queue.append((r, c))
            visited[r][c] = True
    
    # BFS to mark all reachable 0 cells from corners
    while queue:
        cr, cc = queue.popleft()
        for dr, dc in directions:
            nr, nc = cr + dr, cc + dc
            if 0 <= nr < rows and 0 <= nc < cols and output[nr][nc] == 0 and not visited[nr][nc]:
                visited[nr][nc] = True
                queue.append((nr, nc))
    
    # Change unmarked 0 cells to 4
    for r in range(rows):
        for c in range(cols):
            if output[r][c] == 0 and not visited[r][c]:
                output[r][c] = 4
    
    return output