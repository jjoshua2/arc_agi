from collections import defaultdict
from typing import List, Tuple

def transform(grid: List[List[int]]) -> List[List[int]]:
    if not grid or not grid[0]:
        return [row[:] for row in grid]
    h = len(grid)
    w = len(grid[0])
    if h == 0 or w == 0:
        return [row[:] for row in grid]
    bg = grid[0][0]

    visited = [[False] * w for _ in range(h)]
    shapes = []

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for i in range(h):
        for j in range(w):
            if grid[i][j] != bg and not visited[i][j]:
                color = grid[i][j]
                positions = []
                min_r = h + 1
                stack = [(i, j)]
                visited[i][j] = True
                positions.append((i, j))
                min_r = min(min_r, i)

                while stack:
                    x, y = stack.pop()
                    for dx, dy in directions:
                        nx = x + dx
                        ny = y + dy
                        if 0 <= nx < h and 0 <= ny < w and not visited[nx][ny] and grid[nx][ny] == color:
                            visited[nx][ny] = True
                            stack.append((nx, ny))
                            positions.append((nx, ny))
                            min_r = min(min_r, nx)

                row_to_cols = defaultdict(list)
                for rr, cc in positions:
                    row_to_cols[rr].append(cc)
                shapes.append((min_r, color, row_to_cols))

    # Sort shapes by min_r
    shapes.sort(key=lambda s: s[0])

    output = [row[:] for row in grid]

    for idx, (min_r, color, row_to_cols) in enumerate(shapes):
        is_left = (idx % 2 == 0)
        for row in row_to_cols:
            my_cols = sorted(set(row_to_cols[row]))
            if not my_cols:
                continue
            segments = []
            curr_l = my_cols[0]
            curr_r = my_cols[0]
            for k in range(1, len(my_cols)):
                if my_cols[k] == curr_r + 1:
                    curr_r = my_cols[k]
                else:
                    segments.append((curr_l, curr_r))
                    curr_l = curr_r = my_cols[k]
            segments.append((curr_l, curr_r))

            for l, r in segments:
                if is_left:
                    new_l = l - 1
                    if new_l >= 0:
                        for j in range(new_l, r):
                            output[row][j] = color
                        output[row][r] = bg
                    else:
                        # Truncate
                        for j in range(0, r):
                            output[row][j] = color
                        if r < w:
                            output[row][r] = bg
                else:  # right
                    new_r = r + 1
                    if new_r < w:
                        for j in range(l + 1, new_r + 1):
                            output[row][j] = color
                        output[row][l] = bg
                    else:
                        # Truncate
                        for j in range(l + 1, w):
                            output[row][j] = color
                        output[row][l] = bg

    return output