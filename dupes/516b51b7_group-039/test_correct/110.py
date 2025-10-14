import numpy as np
from collections import deque

def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    if not grid_lst:
        return []
    grid = np.array(grid_lst, dtype=int)
    rows, cols = grid.shape
    visited = np.zeros((rows, cols), dtype=bool)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    output = grid.copy()
    for i in range(rows):
        for j in range(cols):
            if grid[i, j] == 1 and not visited[i, j]:
                # Flood fill to find component
                q = deque([(i, j)])
                visited[i, j] = True
                min_r, max_r = i, i
                min_c, max_c = j, j
                while q:
                    r, c = q.popleft()
                    min_r = min(min_r, r)
                    max_r = max(max_r, r)
                    min_c = min(min_c, c)
                    max_c = max(max_c, c)
                    for dr, dc in directions:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] == 1 and not visited[nr, nc]:
                            visited[nr, nc] = True
                            q.append((nr, nc))
                # Verify it's a solid rectangle
                is_rect = True
                h_o = max_r - min_r + 1
                w_o = max_c - min_c + 1
                for rr in range(min_r, max_r + 1):
                    for cc in range(min_c, max_c + 1):
                        if grid[rr, cc] != 1:
                            is_rect = False
                            break
                    if not is_rect:
                        break
                if not is_rect:
                    continue
                h_i = h_o - 2
                w_i = w_o - 2
                if h_i <= 0 or w_i <= 0:
                    continue
                # Set inner to 2
                for rr in range(min_r + 1, max_r):
                    for cc in range(min_c + 1, max_c):
                        output[rr, cc] = 2
                # Now 3s if applicable
                if h_i < 3 or w_i < 3:
                    continue
                c = h_i - 2
                if c <= 0:
                    continue
                central_start = 1  # 0-based inner_r
                central_end = h_i - 2
                left_leg = 1
                right_leg = w_i - 2
                for inner_r in range(central_start, central_end + 1):
                    grid_r = min_r + 1 + inner_r
                    if inner_r == central_start or inner_r == central_end:
                        # full horizontal
                        for inner_c in range(1, w_i - 1):
                            grid_c = min_c + 1 + inner_c
                            output[grid_r, grid_c] = 3
                    else:
                        # vert only
                        # left leg
                        grid_c = min_c + 1 + left_leg
                        output[grid_r, grid_c] = 3
                        # right leg if different
                        if left_leg != right_leg:
                            grid_c = min_c + 1 + right_leg
                            output[grid_r, grid_c] = 3
    return output.tolist()