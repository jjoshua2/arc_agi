import numpy as np
from collections import deque
from typing import List, Tuple

def flood_fill(grid: np.ndarray, sr: int, sc: int, color: int, visited: np.ndarray) -> List[Tuple[int, int]]:
    rows, cols = grid.shape
    queue = deque([(sr, sc)])
    visited[sr, sc] = True
    component = [(sr, sc)]
    dirs = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    while queue:
        r, c = queue.popleft()
        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and not visited[nr, nc] and grid[nr, nc] == color:
                visited[nr, nc] = True
                queue.append((nr, nc))
                component.append((nr, nc))
    return component

def transform(grid_lst: List[List[int]]) -> List[List[int]]:
    if not grid_lst or not grid_lst[0]:
        return []
    grid = np.array(grid_lst, dtype=int)
    rows, cols = grid.shape
    visited = np.zeros((rows, cols), dtype=bool)

    # Find mask from color 8
    mask = None
    for i in range(rows):
        for j in range(cols):
            if grid[i, j] == 8 and not visited[i, j]:
                component = flood_fill(grid, i, j, 8, visited)
                rs = [p[0] for p in component]
                cs = [p[1] for p in component]
                min_r = min(rs)
                min_c = min(cs)
                max_r = max(rs)
                max_c = max(cs)
                h = max_r - min_r + 1
                w = max_c - min_c + 1
                if h == 3 and w == 3:
                    mask = np.zeros((3, 3), dtype=bool)
                    for r, c in component:
                        mask[r - min_r, c - min_c] = True
                    break  # assume one such mask

    if mask is None:
        # Default mask: standard hollow 3x3 with center hole
        mask = np.ones((3, 3), dtype=bool)
        mask[1, 1] = False
        H_m, W_m = 3, 3
    else:
        H_m, W_m = mask.shape

    # Find other components
    for i in range(rows):
        for j in range(cols):
            if grid[i, j] != 0 and grid[i, j] != 8 and not visited[i, j]:
                C = grid[i, j]
                component = flood_fill(grid, i, j, C, visited)
                if not component:
                    continue
                rs = [p[0] for p in component]
                cs = [p[1] for p in component]
                min_r = min(rs)
                min_c = min(cs)
                max_r = max(rs)
                max_c = max(cs)
                h = max_r - min_r + 1
                w = max_c - min_c + 1
                num_cells = len(component)

                # Check if straight line (consecutive no gaps)
                is_vertical = (w == 1 and num_cells == h)
                is_horizontal = (h == 1 and num_cells == w)

                if is_vertical and is_horizontal:  # single cell
                    is_horizontal = True  # default to horizontal expansion
                if not is_vertical and not is_horizontal:
                    continue  # skip non-lines for now

                L = h if is_vertical else w
                base_r = min_r
                base_c = min_c

                if is_vertical:
                    # Expand horizontal, fixed rows base_r to base_r + H_m - 1
                    fixed_start_r = base_r
                    fixed_end_r = fixed_start_r + H_m
                    if fixed_end_r > rows:
                        continue
                    for ii in range(L):
                        start_c = base_c + 4 * ii
                        if start_c + W_m > cols:
                            break
                        for dr in range(H_m):
                            r = fixed_start_r + dr
                            if r >= rows:
                                break
                            for dc in range(W_m):
                                c = start_c + dc
                                if c >= cols:
                                    break
                                if mask[dr, dc] and grid[r, c] == 0:
                                    grid[r, c] = C
                else:
                    # Expand vertical, fixed cols base_c to base_c + W_m - 1
                    fixed_start_c = base_c
                    fixed_end_c = fixed_start_c + W_m
                    if fixed_end_c > cols:
                        continue
                    for ii in range(L):
                        start_r = base_r + 4 * ii
                        if start_r + H_m > rows:
                            break
                        for dr in range(H_m):
                            r = start_r + dr
                            if r >= rows:
                                break
                            for dc in range(W_m):
                                c = fixed_start_c + dc
                                if c >= cols:
                                    break
                                if mask[dr, dc] and grid[r, c] == 0:
                                    grid[r, c] = C

    return grid.tolist()