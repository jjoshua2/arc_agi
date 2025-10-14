import numpy as np

def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    if not grid_lst or not grid_lst[0]:
        return []
    grid = np.array(grid_lst)
    h, w = grid.shape

    # Step 1: Repeated gap filling
    changed = True
    while changed:
        changed = False
        # Horizontal gaps
        for r in range(h):
            for c in range(1, w - 1):
                if grid[r, c] == 5 and grid[r, c - 1] in [2, 8] and grid[r, c + 1] in [2, 8]:
                    grid[r, c] = 8
                    changed = True
        # Vertical gaps
        for c in range(w):
            for r in range(1, h - 1):
                if grid[r, c] == 5 and grid[r - 1, c] in [2, 8] and grid[r + 1, c] in [2, 8]:
                    grid[r, c] = 8
                    changed = True

    # Step 2: Find components of 2 or 8
    visited = np.zeros((h, w), dtype=bool)
    components = []
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for i in range(h):
        for j in range(w):
            if not visited[i, j] and grid[i, j] in [2, 8]:
                comp = []
                stack = [(i, j)]
                visited[i, j] = True
                while stack:
                    cr, cc = stack.pop()
                    comp.append((cr, cc))
                    for dr, dc in directions:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and grid[nr, nc] in [2, 8]:
                            visited[nr, nc] = True
                            stack.append((nr, nc))
                components.append(comp)

    # Step 3: Process each component
    for comp in components:
        if len(comp) < 2:
            continue
        potential_centers = comp  # All cells in comp
        candidates = []
        min_max_l = float('inf')
        for centr, centc in potential_centers:
            # Compute arm lengths
            l_up = 0
            for d in range(1, h):
                nr = centr - d
                if nr < 0 or grid[nr, centc] not in [2, 8]:
                    break
                l_up = d
            l_down = 0
            for d in range(1, h):
                nr = centr + d
                if nr >= h or grid[nr, centc] not in [2, 8]:
                    break
                l_down = d
            l_left = 0
            for d in range(1, w):
                nc = centc - d
                if nc < 0 or grid[centr, nc] not in [2, 8]:
                    break
                l_left = d
            l_right = 0
            for d in range(1, w):
                nc = centc + d
                if nc >= w or grid[centr, nc] not in [2, 8]:
                    break
                l_right = d
            # Check if covers all
            covered = True
            for rr, cc in comp:
                if rr == centr and cc == centc:
                    continue
                if rr == centr:
                    diff = abs(cc - centc)
                    if cc < centc and diff <= l_left:
                        continue
                    if cc > centc and diff <= l_right:
                        continue
                elif cc == centc:
                    diff = abs(rr - centr)
                    if rr < centr and diff <= l_up:
                        continue
                    if rr > centr and diff <= l_down:
                        continue
                covered = False
                break
            if not covered:
                continue
            this_max_l = max(l_up, l_down, l_left, l_right)
            sum_l = l_up + l_down + l_left + l_right
            candidates.append((this_max_l, -sum_l, centr, centc))  # -sum for max sum

        if not candidates:
            continue
        # Pick the one with min max_l, then max sum_l
        candidates.sort()
        _, _, r, c = candidates[0]
        l = candidates[0][0]  # the min max_l

        # Fill
        # up
        for d in range(1, l + 1):
            nr = r - d
            if nr >= 0 and grid[nr, c] == 5:
                grid[nr, c] = 8
        # down
        for d in range(1, l + 1):
            nr = r + d
            if nr < h and grid[nr, c] == 5:
                grid[nr, c] = 8
        # left
        for d in range(1, l + 1):
            nc = c - d
            if nc >= 0 and grid[r, nc] == 5:
                grid[r, nc] = 8
        # right
        for d in range(1, l + 1):
            nc = c + d
            if nc < w and grid[r, nc] == 5:
                grid[r, nc] = 8

    return grid.tolist()