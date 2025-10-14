from collections import deque
from typing import List, Tuple

def transform(grid_lst: List[List[int]]) -> List[List[int]]:
    if not grid_lst or not grid_lst[0]:
        return grid_lst

    rows = len(grid_lst)
    cols = len(grid_lst[0])

    # Copy input to avoid mutating original
    grid = [row[:] for row in grid_lst]

    # Find the most frequent non-zero color -> treat as the wall/line color
    freq = {}
    for r in range(rows):
        for c in range(cols):
            v = grid[r][c]
            if v != 0:
                freq[v] = freq.get(v, 0) + 1

    # If there are no non-zero colors, nothing to preserve; still proceed
    if freq:
        line_color = max(freq.items(), key=lambda x: x[1])[0]
    else:
        line_color = None

    # Find all 4-connected components of background (value == 0)
    visited = [[False] * cols for _ in range(rows)]
    components: List[List[Tuple[int,int]]] = []
    directions = [(1,0),(-1,0),(0,1),(0,-1)]

    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == 0 and not visited[i][j]:
                comp = []
                dq = deque()
                dq.append((i,j))
                visited[i][j] = True
                while dq:
                    x,y = dq.popleft()
                    comp.append((x,y))
                    for dx,dy in directions:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < rows and 0 <= ny < cols:
                            if not visited[nx][ny] and grid[nx][ny] == 0:
                                visited[nx][ny] = True
                                dq.append((nx, ny))
                components.append(comp)

    # If no background components, return grid copy
    if not components:
        return [row[:] for row in grid]

    # Determine largest component
    largest = max(components, key=len)

    # Build output: change only background cells (0)
    out = [row[:] for row in grid]
    for comp in components:
        # largest component -> color 4, others -> color 3
        fill_color = 4 if comp is largest else 3
        for (r,c) in comp:
            out[r][c] = fill_color

    return out