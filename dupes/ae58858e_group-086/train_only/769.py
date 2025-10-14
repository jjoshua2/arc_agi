from typing import List
from collections import deque

def transform(grid_lst: List[List[int]]) -> List[List[int]]:
    if not grid_lst or not grid_lst[0]:
        return [row[:] for row in grid_lst]
    
    rows = len(grid_lst)
    cols = len(grid_lst[0])
    grid = [row[:] for row in grid_lst]
    visited = [[False] * cols for _ in range(rows)]
    
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    def is_valid(r: int, c: int) -> bool:
        return 0 <= r < rows and 0 <= c < cols
    
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 2 and not visited[r][c]:
                # BFS to find component
                component = []
                queue = deque([(r, c)])
                visited[r][c] = True
                while queue:
                    cr, cc = queue.popleft()
                    component.append((cr, cc))
                    for dr, dc in directions:
                        nr, nc = cr + dr, cc + dc
                        if is_valid(nr, nc) and not visited[nr][nc] and grid[nr][nc] == 2:
                            visited[nr][nc] = True
                            queue.append((nr, nc))
                
                # If component size >= 4, change to 6
                if len(component) >= 4:
                    for fr, fc in component:
                        grid[fr][fc] = 6
    
    return grid