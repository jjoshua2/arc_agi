def transform(grid: list[list[int]]) -> list[list[int]]:
    if not grid or not grid[0]:
        return []
    rows = len(grid)
    cols = len(grid[0])
    
    # Find sep_row: row with longest consecutive 5s
    max_consec = 0
    sep_row = -1
    for i in range(rows):
        current_consec = 0
        max_in_row = 0
        for j in range(cols):
            if grid[i][j] == 5:
                current_consec += 1
                if current_consec > max_in_row:
                    max_in_row = current_consec
            else:
                current_consec = 0
        if max_in_row > max_consec:
            max_consec = max_in_row
            sep_row = i
    if sep_row == -1:
        sep_row = rows  # No separator, all top
    
    # Directions for flood fill
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    def get_components(condition_func):
        visited = [[False] * cols for _ in range(rows)]
        components = []
        for i in range(rows):
            for j in range(cols):
                if not visited[i][j] and condition_func(i, j):
                    comp = []
                    stack = [(i, j)]
                    visited[i][j] = True
                    while stack:
                        x, y = stack.pop()
                        comp.append((x, y))
                        for dx, dy in dirs:
                            nx = x + dx
                            ny = y + dy
                            if 0 <= nx < rows and 0 <= ny < cols and not visited[nx][ny] and condition_func(nx, ny):
                                visited[nx][ny] = True
                                stack.append((nx, ny))
                    if comp:
                        components.append(comp)
        return components
    
    # Stamps condition
    def stamp_condition(i, j):
        return i < sep_row and 1 <= grid[i][j] <= 9 and grid[i][j] != 3 and grid[i][j] != 5
    
    stamp_comps = get_components(stamp_condition)
    stamps = []
    for comp in stamp_comps:
        if comp:
            cc = grid[comp[0][0]][comp[0][1]]
            stamps.append({'color': cc, 'positions': comp})
    
    # Canvas condition
    def canvas_condition(i, j):
        return grid[i][j] == 3
    
    canvas_comps = get_components(canvas_condition)
    
    out = [row[:] for row in grid]
    
    for S_list in canvas_comps:
        s = len(S_list)
        if s == 0:
            continue
        S_set = set(S_list)
        possible = []
        for st in stamps:
            c = st['color']
            T_abs = st['positions']
            t = len(T_abs)
            if t < s:
                continue
            # Relative to first point
            if not T_abs:
                continue
            r0, c0 = T_abs[0]
            orig_rel = [(r - r0, c - c0) for r, c in T_abs]
            has_placement = False
            for do_reflect in [False, True]:
                ref_rel = [(dr, -dc if do_reflect else dc) for dr, dc in orig_rel]
                for rot_k in range(4):
                    curr_rel = ref_rel[:]
                    for _ in range(rot_k):
                        curr_rel = [(dc2, -dr2) for dr2, dc2 in curr_rel]
                    rel_set = set(curr_rel)
                    found = False
                    for sr, sc in S_list:
                        for tr, tc in rel_set:
                            vr = sr - tr
                            vc = sc - tc
                            # Check coverage
                            covered = True
                            for ssr, ssc in S_list:
                                ttr = ssr - vr
                                ttc = ssc - vc
                                if (ttr, ttc) not in rel_set:
                                    covered = False
                                    break
                            if not covered:
                                continue
                            # Check extras
                            valid = True
                            for tr2, tc2 in rel_set:
                                pr = vr + tr2
                                pc = vc + tc2
                                if not (0 <= pr < rows and 0 <= pc < cols):
                                    valid = False
                                    break
                                if (pr, pc) not in S_set:
                                    if grid[pr][pc] != 0:
                                        valid = False
                                        break
                            if valid:
                                found = True
                                break
                        if found:
                            break
                    if found:
                        has_placement = True
                        break
                if has_placement:
                    break
            if has_placement:
                possible.append((c, t - s))
        if possible:
            min_e = min(e for _, e in possible)
            cands = [cc for cc, e in possible if e == min_e]
            chosen_c = min(cands)
            for pr, pc in S_list:
                out[pr][pc] = chosen_c
    
    return out