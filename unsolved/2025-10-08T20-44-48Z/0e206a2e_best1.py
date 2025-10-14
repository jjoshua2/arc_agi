import numpy as np
from collections import deque, defaultdict
from itertools import product

def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    if not grid_lst or not grid_lst[0]:
        return []
    grid = np.array(grid_lst)
    rows, cols = grid.shape
    # Find C
    counts = np.bincount(grid.flatten(), minlength=10)[1:]
    C = np.argmax(counts) + 1
    output = grid.copy().astype(int)
    # Directions
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    # Components of C
    visited = np.zeros((rows, cols), dtype=bool)
    components = []
    for i in range(rows):
        for j in range(cols):
            if grid[i, j] == C and not visited[i, j]:
                comp = []
                q = deque([(i, j)])
                visited[i, j] = True
                while q:
                    x, y = q.popleft()
                    comp.append((x, y))
                    for dx, dy in dirs:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < rows and 0 <= ny < cols and grid[nx, ny] == C and not visited[nx, ny]:
                            visited[nx, ny] = True
                            q.append((nx, ny))
                components.append(comp)
    # Attached
    attached_list = []
    attached_visited = np.zeros((rows, cols), dtype=bool)
    for comp in components:
        attached = defaultdict(list)
        for x, y in comp:
            for dx, dy in dirs:
                nx, ny = x + dx, y + dy
                if 0 <= nx < rows and 0 <= ny < cols and not attached_visited[nx, ny] and grid[nx, ny] != 0 and grid[nx, ny] != C:
                    col = grid[nx, ny]
                    attached[col].append((nx, ny))
                    attached_visited[nx, ny] = True
        attached_list.append(attached)
    # Erase
    for i in range(rows):
        for j in range(cols):
            if grid[i, j] == C or attached_visited[i, j]:
                output[i, j] = 0
    # Preserved
    pres_by_col = defaultdict(list)
    for i in range(rows):
        for j in range(cols):
            if output[i, j] != 0:
                pres_by_col[output[i, j]].append((i, j))
    pres_by_col = {k: list(v) for k, v in pres_by_col.items()}
    # Main reconstructions for comps with >=2 attached
    for comp_idx in range(len(components)):
        attached = attached_list[comp_idx]
        colors = list(attached.keys())
        num = len(colors)
        if num < 2:
            continue
        orig_att_pos = {col: attached[col][0] for col in colors}
        avail = [pres_by_col.get(col, []) for col in colors]
        if any(len(l) == 0 for l in avail):
            continue
        valid_assigns = []
        for assign_tup in product(*avail):
            assign_pos = dict(zip(colors, assign_tup))
            # Diameter
            poss = [assign_pos[col] for col in colors]
            diam = max(abs(p1[0] - p2[0]) + abs(p1[1] - p2[1]) for p1 in poss for p2 in poss)
            # Check isometries
            ref_col = colors[0]
            orig_ref = orig_att_pos[ref_col]
            pres_ref = assign_pos[ref_col]
            rel_orig = {col: (orig_att_pos[col][0] - orig_ref[0], orig_att_pos[col][1] - orig_ref[1]) for col in colors}
            rel_pres = {col: (assign_pos[col][0] - pres_ref[0], assign_pos[col][1] - pres_ref[1]) for col in colors}
            matching_isoms = []
            for rot in range(4):
                for fr in [0, 1]:
                    for fc in [0, 1]:
                        match_this = True
                        for col in colors:
                            if col == ref_col:
                                continue
                            odx, ody = rel_orig[col]
                            # rot
                            if rot == 0:
                                rdx, rdy = odx, ody
                            elif rot == 1:
                                rdx, rdy = -ody, odx
                            elif rot == 2:
                                rdx, rdy = -odx, -ody
                            elif rot == 3:
                                rdx, rdy = ody, -odx
                            if fr:
                                rdx = -rdx
                            if fc:
                                rdy = -rdy
                            pdx, pdy = rel_pres[col]
                            if rdx != pdx or rdy != pdy:
                                match_this = False
                                break
                        if match_this:
                            matching_isoms.append((rot, fr, fc))
            if matching_isoms:
                # Best isom for this assign
                local_best_can = -1
                local_best_sum = -1
                local_best_isom = None
                local_best_places = None
                for rot, fr, fc in matching_isoms:
                    places = []
                    s_nx = 0
                    can = 0
                    for cx, cy in components[comp_idx]:
                        dx = cx - orig_ref[0]
                        dy = cy - orig_ref[1]
                        if rot == 0:
                            rdx, rdy = dx, dy
                        elif rot == 1:
                            rdx, rdy = -dy, dx
                        elif rot == 2:
                            rdx, rdy = -dx, -dy
                        elif rot == 3:
                            rdx, rdy = dy, -dx
                        if fr:
                            rdx = -rdx
                        if fc:
                            rdy = -rdy
                        nx = pres_ref[0] + rdx
                        ny = pres_ref[1] + rdy
                        if 0 <= nx < rows and 0 <= ny < cols and output[nx, ny] == 0:
                            places.append((nx, ny))
                            s_nx += nx
                            can += 1
                    if can > local_best_can or (can == local_best_can and s_nx > local_best_sum):
                        local_best_can = can
                        local_best_sum = s_nx
                        local_best_isom = (rot, fr, fc)
                        local_best_places = places
                if local_best_can >= 1:  # at least some
                    valid_assigns.append((diam, -local_best_can, -local_best_sum, assign_pos, local_best_isom, local_best_places))
        if valid_assigns:
            valid_assigns.sort()
            _, _, _, best_assign, best_isom, best_places = valid_assigns[0]
            for nx, ny in best_places:
                output[nx, ny] = C
            for col in colors:
                pres_by_col[col] = [p for p in pres_by_col[col] if p != best_assign[col]]
    # Extra singles
    had_attached_types = set()
    for att in attached_list:
        had_attached_types.update(att.keys())
    for col in list(had_attached_types):
        while col in pres_by_col and pres_by_col[col]:
            pres_pos = pres_by_col[col].pop(0)
            # Find comp_idx and orig_pos
            comp_idx = -1
            orig_pos = None
            for ci, att in enumerate(attached_list):
                if col in att and att[col]:
                    comp_idx = ci
                    orig_pos = att[col][0]
                    break
            if comp_idx == -1 or orig_pos is None:
                continue
            # Adjacent rel
            adj_rel = []
            ox, oy = orig_pos
            for cx, cy in components[comp_idx]:
                if (abs(cx - ox) + abs(cy - oy) == 1):
                    dx = cx - ox
                    dy = cy - oy
                    adj_rel.append((dx, dy))
            # Options
            options = [False, True]
            best_can = -1
            best_sum_nx = -1
            best_places = None
            for flip in options:
                places = []
                s_nx = 0
                can = 0
                for dx, dy in adj_rel:
                    tdx = -dx if flip else dx
                    tdy = dy
                    nx = pres_pos[0] + tdx
                    ny = pres_pos[1] + tdy
                    if 0 <= nx < rows and 0 <= ny < cols and output[nx, ny] == 0:
                        places.append((nx, ny))
                        s_nx += nx
                        can += 1
                if can > best_can or (can == best_can and s_nx > best_sum_nx):
                    best_can = can
                    best_sum_nx = s_nx
                    best_places = places
            for nx, ny in best_places:
                output[nx, ny] = C
    return output.tolist()