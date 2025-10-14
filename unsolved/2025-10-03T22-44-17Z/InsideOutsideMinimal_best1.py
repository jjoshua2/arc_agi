from collections import deque
from typing import List

def transform(grid_lst: List[List[int]]) -> List[List[int]]:
    if not grid_lst:
        return []
    rows = len(grid_lst)
    if rows == 0:
        return []
    cols = len(grid_lst[0])
    input_grid = grid_lst
    output = [row[:] for row in input_grid]
    visited = [[False] * cols for _ in range(rows)]
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def bfs(start_r: int, start_c: int, color: int) -> List[tuple[int, int]]:
        queue = deque([(start_r, start_c)])
        visited[start_r][start_c] = True
        component = [(start_r, start_c)]
        while queue:
            r, c = queue.popleft()
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc] and input_grid[nr][nc] == color:
                    visited[nr][nc] = True
                    queue.append((nr, nc))
                    component.append((nr, nc))
        return component

    for i in range(rows):
        for j in range(cols):
            if input_grid[i][j] != 0 and not visited[i][j]:
                color = input_grid[i][j]
                component = bfs(i, j, color)
                rs = [pos[0] for pos in component]
                cs = [pos[1] for pos in component]
                min_r = min(rs)
                max_r = max(rs)
                min_c = min(cs)
                max_c = max(cs)
                h = max_r - min_r + 1
                w = max_c - min_c + 1
                perimeter = 2 * (h + w) - 4
                if h >= 3 and w >= 3 and len(component) == perimeter:
                    # Clear interior
                    for ir in range(min_r + 1, max_r):
                        for ic in range(min_c + 1, max_c):
                            output[ir][ic] = 0
    return output