from collections import deque
import copy

def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    if not grid_lst or not grid_lst[0]:
        return copy.deepcopy(grid_lst)
    
    grid = copy.deepcopy(grid_lst)
    rows = len(grid)
    cols = len(grid[0])
    
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] != 0 and grid[i][j] != 5:
                color = grid[i][j]
                q = deque([(i, j)])
                while q:
                    r, c = q.popleft()
                    for dr, dc in directions:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 0:
                            grid[nr][nc] = color
                            q.append((nr, nc))
    
    return grid