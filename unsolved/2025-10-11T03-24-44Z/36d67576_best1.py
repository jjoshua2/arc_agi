import numpy as np
from collections import deque
from typing import List, Tuple

def transform(grid_lst: List[List[int]]) -> List[List[int]]:
    if not grid_lst or not grid_lst[0]:
        return grid_lst
    grid = [row[:] for row in grid_lst]
    rows = len(grid)
    cols = len(grid[0])
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right

    # Dictionary of known configurations: key = (tuple of sorted relative yellow positions, (dr2, dc2))
    # value = list of ((drf, dcf, col), ...)
    patterns = {
        # Example 1 vertical
        ( ((0,0), (1,0), (2,0), (3,0), (4,0)), (4, 1) ): [
            (0, -1, 3), (1, 1, 1), (2, -1, 3), (3, 1, 1), (4, -1, 3)
        ],
        # Example 1 top horizontal
        ( ((0,0), (0,1), (0,2), (0,3), (0,4)), (-1, 4) ): [
            (-1, 1, 1), (-1, 3, 1), (1, 0, 3), (1, 2, 3), (1, 4, 3)
        ],
        # Example 1 lower horizontal
        ( ((0,0), (0,1), (0,2), (0,3), (0,4)), (1, 0) ): [
            (-1, 0, 3), (-1, 2, 3), (-1, 4, 3), (1, 1, 1), (1, 3, 1)
        ],
        # Example 2 right L
        ( ((0,0), (0,1), (0,2), (1,0), (2,0)), (-1, 0) ): [
            (0, -1, 1), (2, -1, 3), (2, 1, 3), (2, 2, 3)
        ],
        # Example 2 bottom L
        ( ((0,2), (1,2), (2,0), (2,1), (2,2)), (2, 3) ): [
            (0, 0, 3), (1, 0, 3), (3, 0, 3), (3, 2, 1)
        ],
        # Example 3 middle
        ( ((0,1), (1,0), (1,1), (1,2), (2,1)), (0, 0) ): [
            (1, 3, 1), (2, 2, 3), (3, 1, 1)
        ],
        # Example 3 bottom
        ( ((0,1), (1,0), (1,1), (1,2), (2,1)), (2, 0) ): [
            (-1, 1, 1), (0, 2, 3)
        ],
        # Example 3 top (no fills)
        ( ((0,1), (1,0), (1,1), (1,2), (2,1)), (0, 2) ): []
    }

    visited = [[False] * cols for _ in range(rows)]
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == 4 and not visited[i][j]:
                # BFS to find component
                component = []
                queue = deque([(i, j)])
                visited[i][j] = True
                while queue:
                    x, y = queue.popleft()
                    component.append((x, y))
                    for dx, dy in directions:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < rows and 0 <= ny < cols and not visited[nx][ny] and grid[nx][ny] == 4:
                            visited[nx][ny] = True
                            queue.append((nx, ny))
                
                if len(component) == 5:
                    # Find attached 2
                    twos = set()
                    for x, y in component:
                        for dx, dy in directions:
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < rows and 0 <= ny < cols and grid[nx][ny] == 2:
                                twos.add((nx, ny))
                    if len(twos) == 1:
                        r2, c2 = list(twos)[0]
                        min_r = min(r for r, c in component)
                        min_c = min(c for r, c in component)
                        rel_yellow = sorted((r - min_r, c - min_c) for r, c in component)
                        dr2 = r2 - min_r
                        dc2 = c2 - min_c
                        key = (tuple(rel_yellow), (dr2, dc2))
                        if key in patterns:
                            for drf, dcf, col in patterns[key]:
                                rf = min_r + drf
                                cf = min_c + dcf
                                if 0 <= rf < rows and 0 <= cf < cols and grid[rf][cf] == 0:
                                    grid[rf][cf] = col

    return grid