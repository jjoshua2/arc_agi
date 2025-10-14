from collections import deque
import copy

def transform(grid: list[list[int]]) -> list[list[int]]:
    if not grid or not grid[0]:
        return []
    rows = len(grid)
    cols = len(grid[0])
    new_grid = copy.deepcopy(grid)
    visited = [[False] * cols for _ in range(rows)]
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    for i in range(rows):
        for j in range(cols):
            if new_grid[i][j] == 0 and not visited[i][j]:
                # BFS to find component
                component = []
                touches_border = False
                queue = deque([(i, j)])
                visited[i][j] = True
                component.append((i, j))
                if i == 0 or i == rows - 1 or j == 0 or j == cols - 1:
                    touches_border = True
                
                while queue:
                    r, c = queue.popleft()
                    for dr, dc in directions:
                        nr = r + dr
                        nc = c + dc
                        if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc] and new_grid[nr][nc] == 0:
                            visited[nr][nc] = True
                            queue.append((nr, nc))
                            component.append((nr, nc))
                            if nr == 0 or nr == rows - 1 or nc == 0 or nc == cols - 1:
                                touches_border = True
                
                # If does not touch any border, fill with 4
                if not touches_border:
                    for cr, cc in component:
                        new_grid[cr][cc] = 4
    
    return new_grid