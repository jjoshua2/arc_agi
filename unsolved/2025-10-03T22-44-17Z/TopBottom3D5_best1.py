import numpy as np
from collections import defaultdict

def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    if not grid_lst:
        return []
    grid = np.array(grid_lst)
    rows, cols = grid.shape
    visited = np.zeros((rows, cols), dtype=bool)
    directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    pattern_count = defaultdict(int)
    for i in range(rows):
        for j in range(cols):
            if grid[i, j] != 0 and not visited[i, j]:
                color = grid[i, j]
                component = []
                stack = [(i, j)]
                visited[i, j] = True
                min_r, max_r = i, i
                min_c, max_c = j, j
                while stack:
                    x, y = stack.pop()
                    component.append((x, y))
                    min_r = min(min_r, x)
                    max_r = max(max_r, x)
                    min_c = min(min_c, y)
                    max_c = max(max_c, y)
                    for dx, dy in directions:
                        nx = x + dx
                        ny = y + dy
                        if 0 <= nx < rows and 0 <= ny < cols and not visited[nx, ny] and grid[nx, ny] == color:
                            visited[nx, ny] = True
                            stack.append((nx, ny))
                h = max_r - min_r + 1
                w = max_c - min_c + 1
                if h == 3 and w == 3 and len(component) > 0:
                    rel_pattern = frozenset((r - min_r, c - min_c) for r, c in component)
                    key = (color, rel_pattern)
                    pattern_count[key] += 1
    if not pattern_count:
        return [[0] * 3 for _ in range(3)]
    max_cnt = max(pattern_count.values())
    best_key = max((k for k, v in pattern_count.items() if v == max_cnt), key=lambda k: k[0])
    color, pat = best_key
    out = [[0] * 3 for _ in range(3)]
    for r, c in pat:
        out[r][c] = color
    return out