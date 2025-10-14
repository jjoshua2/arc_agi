from collections import deque
import copy

def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    if not grid_lst or not grid_lst[0]:
        return []
    rows = len(grid_lst)
    cols = len(grid_lst[0])
    grid = grid_lst  # no need to copy grid, we'll copy for output
    output = [row[:] for row in grid_lst]
    
    # Flood fill to find all exterior 0s
    exterior = set()
    visited = [[False] * cols for _ in range(rows)]
    q = deque()
    
    # Add border 0s to queue
    for i in range(rows):
        if grid[i][0] == 0 and not visited[i][0]:
            visited[i][0] = True
            q.append((i, 0))
            exterior.add((i, 0))
        if grid[i][cols - 1] == 0 and not visited[i][cols - 1]:
            visited[i][cols - 1] = True
            q.append((i, cols - 1))
            exterior.add((i, cols - 1))
    for j in range(cols):
        if grid[0][j] == 0 and not visited[0][j]:
            visited[0][j] = True
            q.append((0, j))
            exterior.add((0, j))
        if grid[rows - 1][j] == 0 and not visited[rows - 1][j]:
            visited[rows - 1][j] = True
            q.append((rows - 1, j))
            exterior.add((rows - 1, j))
    
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    while q:
        r, c = q.popleft()
        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc] and grid[nr][nc] == 0:
                visited[nr][nc] = True
                q.append((nr, nc))
                exterior.add((nr, nc))
    
    # Now find 1-components
    visited_1 = [[False] * cols for _ in range(rows)]
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == 1 and not visited_1[i][j]:
                # New component
                component = []
                comp_q = deque([(i, j)])
                visited_1[i][j] = True
                component.append((i, j))
                has_interior = False
                
                while comp_q:
                    r, c = comp_q.popleft()
                    # Check neighbors for interior 0
                    for dr, dc in dirs:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < rows and 0 <= nc < cols:
                            if grid[nr][nc] == 0 and (nr, nc) not in exterior:
                                has_interior = True
                            elif grid[nr][nc] == 1 and not visited_1[nr][nc]:
                                visited_1[nr][nc] = True
                                comp_q.append((nr, nc))
                                component.append((nr, nc))
                
                # If the component encloses an interior 0, change to 3
                if has_interior:
                    for cr, cc in component:
                        output[cr][cc] = 3
    
    return output