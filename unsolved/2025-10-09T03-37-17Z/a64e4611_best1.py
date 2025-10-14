from collections import deque
from typing import List

def transform(grid: List[List[int]]) -> List[List[int]]:
    if not grid or not grid[0]:
        return []
    
    rows = len(grid)
    cols = len(grid[0])
    output = [row[:] for row in grid]
    visited = [[False] * cols for _ in range(rows)]
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 4-connectivity
    
    def bfs(r: int, c: int) -> List[tuple]:
        queue = deque([(r, c)])
        component = []
        visited[r][c] = True
        while queue:
            cr, cc = queue.popleft()
            component.append((cr, cc))
            for dr, dc in directions:
                nr, nc = cr + dr, cc + dc
                if (0 <= nr < rows and 0 <= nc < cols and
                    not visited[nr][nc] and grid[nr][nc] == 0):
                    visited[nr][nc] = True
                    queue.append((nr, nc))
        return component
    
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 0 and not visited[r][c]:
                component = bfs(r, c)
                touches_border = any(
                    rr == 0 or rr == rows - 1 or cc == 0 or cc == cols - 1
                    for rr, cc in component
                )
                if not touches_border:
                    for rr, cc in component:
                        output[rr][cc] = 3
    
    return output