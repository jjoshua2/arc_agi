from collections import deque, Counter, defaultdict
from typing import List

def transform(grid: List[List[int]]) -> List[List[int]]:
    if not grid or not grid[0]:
        return []
    h = len(grid)
    w = len(grid[0])
    color_count = defaultdict(int)
    for i in range(h):
        for j in range(w):
            val = grid[i][j]
            if val != 0:
                color_count[val] += 1
    out = [row[:] for row in grid]
    visited = [[False] * w for _ in range(h)]
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for i in range(h):
        for j in range(w):
            if not visited[i][j] and grid[i][j] != 0:
                col = grid[i][j]
                q = deque([(i, j)])
                visited[i][j] = True
                component = [(i, j)]
                while q:
                    x, y = q.popleft()
                    for dx, dy in directions:
                        nx = x + dx
                        ny = y + dy
                        if 0 <= nx < h and 0 <= ny < w and not visited[nx][ny] and grid[nx][ny] == col:
                            visited[nx][ny] = True
                            q.append((nx, ny))
                            component.append((nx, ny))
                size = len(component)
                if size >= 3:
                    continue
                elif size == 2:
                    for px, py in component:
                        out[px][py] = 0
                elif size == 1:
                    ii, jj = component[0]
                    S = grid[ii][jj]
                    # Get neighbor colors
                    left_c = grid[ii][jj - 1] if jj > 0 else 0
                    right_c = grid[ii][jj + 1] if jj + 1 < w else 0
                    up_c = grid[ii - 1][jj] if ii > 0 else 0
                    down_c = grid[ii + 1][jj] if ii + 1 < h else 0
                    neigh_colors = [left_c, right_c, up_c, down_c]
                    count = Counter(neigh_colors)
                    set_to = None
                    if S != 8:
                        candidates = []
                        for CC in count:
                            cnt = count[CC]
                            if CC != 0 and CC != 8 and cnt >= 2:
                                candidates.append((color_count[CC], CC))
                        if candidates:
                            candidates.sort(key=lambda x: x[0], reverse=True)
                            set_to = candidates[0][1]
                    else:
                        # Check horizontal
                        if jj > 0 and jj + 1 < w and left_c == right_c and left_c != 0 and left_c != 8:
                            set_to = left_c
                        # Check vertical
                        if set_to is None and ii > 0 and ii + 1 < h and up_c == down_c and up_c != 0 and up_c != 8:
                            set_to = up_c
                    out[ii][jj] = set_to if set_to is not None else 0
    return out