from typing import List, Tuple

def find_components(grid: List[List[int]], is_zero: bool = False) -> List[List[Tuple[int, int]]]:
    if not grid or not grid[0]:
        return []
    rows = len(grid)
    cols = len(grid[0])
    visited = set()
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    components = []
    for r in range(rows):
        for c in range(cols):
            if (r, c) not in visited:
                cell_val = grid[r][c]
                if (is_zero and cell_val == 0) or (not is_zero and cell_val != 0):
                    component = []
                    stack = [(r, c)]
                    visited.add((r, c))
                    while stack:
                        x, y = stack.pop()
                        component.append((x, y))
                        for dx, dy in directions:
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < rows and 0 <= ny < cols and (nx, ny) not in visited:
                                nval = grid[nx][ny]
                                if (is_zero and nval == 0) or (not is_zero and nval != 0):
                                    visited.add((nx, ny))
                                    stack.append((nx, ny))
                    components.append(component)
    return components

def transform(grid: List[List[int]]) -> List[List[int]]:
    if not grid or not grid[0]:
        return []
    non_zero_comps = find_components(grid, False)
    if not non_zero_comps:
        return [[]]
    comp_sizes = [len(comp) for comp in non_zero_comps]
    max_size_idx = comp_sizes.index(max(comp_sizes))
    main_comp = non_zero_comps[max_size_idx]
    small_comps = [comp for idx, comp in enumerate(non_zero_comps) if idx != max_size_idx]

    # Main bounding box
    main_min_r = min(r for r, c in main_comp)
    main_max_r = max(r for r, c in main_comp)
    main_min_c = min(c for r, c in main_comp)
    main_max_c = max(c for r, c in main_comp)
    h_out = main_max_r - main_min_r + 1
    w_out = main_max_c - main_min_c + 1
    out = [[2 for _ in range(w_out)] for _ in range(h_out)]

    # Find hole components: zero components inside main bb and not touching border
    border_rows = {main_min_r, main_max_r}
    border_cols = {main_min_c, main_max_c}
    zero_comps = find_components(grid, True)
    hole_comps = []
    for zcomp in zero_comps:
        if not zcomp:
            continue
        z_min_r = min(r for r, c in zcomp)
        z_max_r = max(r for r, c in zcomp)
        z_min_c = min(c for r, c in zcomp)
        z_max_c = max(c for r, c in zcomp)
        if not (main_min_r <= z_min_r and z_max_r <= main_max_r and main_min_c <= z_min_c and z_max_c <= main_max_c):
            continue
        touches_border = any(r in border_rows or c in border_cols for r, c in zcomp)
        if touches_border:
            continue
        hole_comps.append(zcomp)

    # Prepare smalls
    smalls = []
    for comp in small_comps:
        if not comp:
            continue
        non2_colors = [grid[r][c] for r, c in comp if grid[r][c] != 2]
        if not non2_colors:
            continue
        paint_c = non2_colors[0]
        if not all(color == paint_c for color in non2_colors):
            # Assume uniform, skip or handle if needed
            continue
        s_min_r = min(r for r, c in comp)
        s_max_r = max(r for r, c in comp)
        s_h = s_max_r - s_min_r + 1
        s_min_c = min(c for r, c in comp)
        s_max_c = max(c for r, c in comp)
        s_w = s_max_c - s_min_c + 1
        smalls.append((s_min_r, s_min_c, s_h, s_w, paint_c))

    # Sort smalls by min_r then min_c
    smalls.sort(key=lambda x: (x[0], x[1]))

    # Prepare holes
    holes = []
    for hcomp in hole_comps:
        if not hcomp:
            continue
        h_min_r = min(r for r, c in hcomp)
        h_min_c = min(c for r, c in hcomp)
        holes.append((h_min_r, h_min_c, hcomp))

    # Sort holes by min_r then min_c
    holes.sort(key=lambda x: (x[0], x[1]))

    # Pair and paint
    num_pairs = min(len(smalls), len(holes))
    for k in range(num_pairs):
        s_min_r, s_min_c, s_h, s_w, paint_c = smalls[k]
        h_min_r, h_min_c, hcomp = holes[k]
        exp_min_r = h_min_r
        exp_max_r = h_min_r + s_h - 1
        exp_min_c = h_min_c
        exp_max_c = h_min_c + s_w - 1
        for tr in range(exp_min_r, exp_max_r + 1):
            for tc in range(exp_min_c, exp_max_c + 1):
                if (main_min_r <= tr <= main_max_r and
                    main_min_c <= tc <= main_max_c and
                    grid[tr][tc] == 2):
                    i = tr - main_min_r
                    j = tc - main_min_c
                    out[i][j] = paint_c

    return out