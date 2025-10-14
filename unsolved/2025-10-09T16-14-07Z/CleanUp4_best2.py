from collections import deque, defaultdict
from typing import List

def transform(grid: List[List[int]]) -> List[List[int]]:
    if not grid or not grid[0]:
        return []
    h = len(grid)
    w = len(grid[0])
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    visited = [[False] * w for _ in range(h)]
    outer_pos = defaultdict(set)
    small_comps = []
    for i in range(h):
        for j in range(w):
            if not visited[i][j] and grid[i][j] != 0:
                color = grid[i][j]
                pos = []
                q = deque([(i, j)])
                visited[i][j] = True
                while q:
                    x, y = q.popleft()
                    pos.append((x, y))
                    for dx, dy in directions:
                        nx = x + dx
                        ny = y + dy
                        if 0 <= nx < h and 0 <= ny < w and not visited[nx][ny] and grid[nx][ny] == color:
                            visited[nx][ny] = True
                            q.append((nx, ny))
                size = len(pos)
                touches_border = any(x == 0 or x == h - 1 or y == 0 or y == w - 1 for x, y in pos)
                if size > 2 and touches_border:
                    for x, y in pos:
                        outer_pos[color].add((x, y))
                if size <= 2:
                    small_comps.append({'pos': pos})
    out = [row[:] for row in grid]
    for sc in small_comps:
        max_links = 0
        best_b = 0
        for b, pos_set in outer_pos.items():
            links = 0
            for x, y in sc['pos']:
                for dx, dy in directions:
                    nx = x + dx
                    ny = y + dy
                    if 0 <= nx < h and 0 <= ny < w and (nx, ny) in pos_set:
                        links += 1
            if links > max_links:
                max_links = links
                best_b = b
        target = best_b if max_links >= 2 else 0
        for x, y in sc['pos']:
            out[x][y] = target
    return out