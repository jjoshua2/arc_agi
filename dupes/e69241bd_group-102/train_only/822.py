from collections import deque
from typing import List

def transform(input_grid: List[List[int]]) -> List[List[int]]:
    if not input_grid or not input_grid[0]:
        return []
    rows = len(input_grid)
    cols = len(input_grid[0])
    grid = [row[:] for row in input_grid]
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] != 0 and grid[i][j] != 5:
                color = grid[i][j]
                q = deque([(i, j)])
                while q:
                    r, c = q.popleft()
                    for dr, dc in directions:
                        nr = r + dr
                        nc = c + dc
                        if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 0:
                            grid[nr][nc] = color
                            q.append((nr, nc))
    return grid