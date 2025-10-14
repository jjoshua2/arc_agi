from collections import defaultdict
from typing import List, Tuple

def get_components(grid: List[List[int]], condition) -> List[List[Tuple[int, int]]]:
    if not grid or not grid[0]:
        return []
    h, w = len(grid), len(grid[0])
    visited = [[False] * w for _ in range(h)]
    components = []
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for i in range(h):
        for j in range(w):
            if not visited[i][j] and condition(i, j):
                comp = []
                stack = [(i, j)]
                visited[i][j] = True
                while stack:
                    x, y = stack.pop()
                    comp.append((x, y))
                    for dx, dy in directions:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < h and 0 <= ny < w and not visited[nx][ny] and condition(nx, ny):
                            visited[nx][ny] = True
                            stack.append((nx, ny))
                components.append(comp)
    return components

def transform(grid: List[List[int]]) -> List[List[int]]:
    if not grid or not grid[0]:
        return []
    h, w = len(grid), len(grid[0])
    output = [row[:] for row in grid]

    def is_stamp(r: int, c: int) -> bool:
        return 0 <= r < h and 0 <= c < w and grid[r][c] != 0 and grid[r][c] != 8

    def is_eight(r: int, c: int) -> bool:
        return 0 <= r < h and 0 <= c < w and grid[r][c] == 8

    stamp_comps = get_components(grid, is_stamp)
    eight_comps = get_components(grid, is_eight)

    if not stamp_comps:
        return output

    stamp_cells = stamp_comps[0]  # Assume single stamp component
    # Clear stamp cells
    for r, c in stamp_cells:
        output[r][c] = 0

    # Process stamp to get rel_colors
    if not stamp_cells:
        return output
    stamp_min_r = min(r for r, _ in stamp_cells)
    stamp_row_dict = defaultdict(list)
    for r, c in stamp_cells:
        stamp_row_dict[r].append((c, grid[r][c]))
    max_stamp_r = max(stamp_row_dict.keys())
    stamp_rel_colors = []
    for rr in range(stamp_min_r, max_stamp_r + 1):
        row_list = stamp_row_dict[rr]
        row_list.sort(key=lambda x: x[0])
        colors = [color for _, color in row_list]
        stamp_rel_colors.append(colors)
    stamp_height = len(stamp_rel_colors)

    # Now process each eight component
    for comp_cells in eight_comps:
        if not comp_cells:
            continue
        comp_row_dict = defaultdict(list)
        for r, c in comp_cells:
            comp_row_dict[r].append(c)
        comp_min_r = min(comp_row_dict.keys())
        comp_max_r = max(comp_row_dict.keys())
        comp_height = comp_max_r - comp_min_r + 1
        for k in range(comp_height):
            r = comp_min_r + k
            comp_cols = sorted(comp_row_dict[r])
            num_cells = len(comp_cols)
            if k < stamp_height:
                pattern = stamp_rel_colors[k]
                # Assume len(pattern) == num_cells
                for i, cc in enumerate(comp_cols):
                    if i < len(pattern):
                        output[r][cc] = pattern[i]
            else:
                # Copy from above
                prev_r = r - 1
                for cc in comp_cols:
                    # Assume output[prev_r][cc] is set
                    output[r][cc] = output[prev_r][cc]

    return output