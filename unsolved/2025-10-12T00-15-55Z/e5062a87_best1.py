from collections import deque
import copy

def transform(grid):
    if not grid or not grid[0]:
        return []
    h = len(grid)
    w = len(grid[0])
    
    # Find the boundary color C (unique non-zero)
    non_zero_colors = set()
    for row in grid:
        for val in row:
            if val != 0:
                non_zero_colors.add(val)
    if not non_zero_colors:
        return copy.deepcopy(grid)
    C = list(non_zero_colors)[0]  # Assume single non-zero color
    
    output = copy.deepcopy(grid)
    visited = [[False] * w for _ in range(h)]
    q = deque()
    
    # Directions for flood fill
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    # Add all border cells that are 0 to queue and mark as -1 (temp for exterior)
    for i in range(h):
        # Left border
        if output[i][0] == 0:
            q.append((i, 0))
            visited[i][0] = True
            output[i][0] = -1
        # Right border
        if output[i][w - 1] == 0:
            q.append((i, w - 1))
            visited[i][w - 1] = True
            output[i][w - 1] = -1
    
    for j in range(w):
        # Top border
        if output[0][j] == 0:
            q.append((0, j))
            visited[0][j] = True
            output[0][j] = -1
        # Bottom border
        if output[h - 1][j] == 0:
            q.append((h - 1, j))
            visited[h - 1][j] = True
            output[h - 1][j] = -1
    
    # Flood fill from borders through 0 cells only (C blocks)
    while q:
        x, y = q.popleft()
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < h and 0 <= ny < w and not visited[nx][ny] and output[nx][ny] == 0:
                visited[nx][ny] = True
                output[nx][ny] = -1
                q.append((nx, ny))
    
    # Restore exterior to 0 and fill interiors (remaining 0s) with 2
    for i in range(h):
        for j in range(w):
            if output[i][j] == -1:
                output[i][j] = 0
            elif output[i][j] == 0:
                output[i][j] = 2
    
    # C cells remain unchanged
    return output