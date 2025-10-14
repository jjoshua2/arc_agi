from collections import deque
from typing import List

def transform(grid: List[List[int]]) -> List[List[int]]:
    if not grid or not grid[0]:
        return grid
    rows = len(grid)
    cols = len(grid[0])
    visited = set()
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 0 and (r, c) not in visited:
                color = grid[r][c]
                component = []
                q = deque([(r, c)])
                visited.add((r, c))
                component.append((r, c))
                
                while q:
                    cr, cc = q.popleft()
                    for dr, dc in directions:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited and grid[nr][nc] == color:
                            visited.add((nr, nc))
                            q.append((nr, nc))
                            component.append((nr, nc))
                
                if len(component) < 3:
                    for pr, pc in component:
                        grid[pr][pc] = 3
    
    return grid