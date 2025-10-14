from collections import deque
from typing import List, Tuple

def transform(grid: List[List[int]]) -> List[List[int]]:
    if not grid or not grid[0]:
        return grid
    rows = len(grid)
    cols = len(grid[0])
    output = [row[:] for row in grid]
    visited = [[False] * cols for _ in range(rows)]
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def is_valid(r: int, c: int) -> bool:
        return 0 <= r < rows and 0 <= c < cols

    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == 0 and not visited[i][j]:
                # Start BFS for new component
                component = []
                queue = deque([(i, j)])
                visited[i][j] = True
                touches_border = (i == 0 or i == rows - 1 or j == 0 or j == cols - 1)
                while queue:
                    r, c = queue.popleft()
                    component.append((r, c))
                    if r == 0 or r == rows - 1 or c == 0 or c == cols - 1:
                        touches_border = True
                    for dr, dc in directions:
                        nr, nc = r + dr, c + dc
                        if is_valid(nr, nc) and grid[nr][nc] == 0 and not visited[nr][nc]:
                            visited[nr][nc] = True
                            queue.append((nr, nc))
                # If doesn't touch border, fill with 4
                if not touches_border:
                    for r, c in component:
                        output[r][c] = 4
    return output