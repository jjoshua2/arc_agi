from typing import List, Tuple

def transform(grid_lst: List[List[int]]) -> List[List[int]]:
    if not grid_lst or not grid_lst[0]:
        return []
    grid = [row[:] for row in grid_lst]
    rows = len(grid)
    cols = len(grid[0])
    D = 4  # yellow blocks we operate on (observed in examples)

    # helper: find connected components of color 'val' using 4-neighbors
    def find_components(val: int):
        visited = [[False] * cols for _ in range(rows)]
        comps = []
        for r in range(rows):
            for c in range(cols):
                if not visited[r][c] and grid[r][c] == val:
                    stack = [(r, c)]
                    visited[r][c] = True
                    comp = []
                    while stack:
                        x, y = stack.pop()
                        comp.append((x, y))
                        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < rows and 0 <= ny < cols and not visited[nx][ny] and grid[nx][ny] == val:
                                visited[nx][ny] = True
                                stack.append((nx, ny))
                    comps.append(sorted(comp))
        return comps

    comps = find_components(D)
    if not comps:
        return grid

    # For each comp compute count of adjacent decoration cells (non-zero, not D)
    def adjacent_decorations(comp):
        decs = []
        seen = set()
        for (r, c) in comp:
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                rr, cc = r+dr, c+dc
                if 0 <= rr < rows and 0 <= cc < cols:
                    val = grid[rr][cc]
                    if val != 0 and val != D:
                        if (rr,cc) not in seen:
                            seen.add((rr,cc))
                            decs.append(((rr,cc), val))
        return decs

    # choose canonical component: max number of adjacent decorations, tie-break by smallest (r,c)
    comp_decor_list = [(comp, adjacent_decorations(comp)) for comp in comps]
    comp_decor_list.sort(key=lambda x: (-len(x[1]), min(x[0])))  # more decorations, then top-left comp
    canonical_comp, canonical_decor_list = comp_decor_list[0]
    # canonical_comp: list of D coords (tuples)
    P = canonical_comp
    P_set = set(P)

    if not canonical_decor_list:
        # nothing to propagate
        return grid

    # flatten canonical decorations: list of (pos,color)
    canonical_decorations = canonical_decor_list  # list of ((r,c), color)

    # Build list of candidate orthonormal transforms (8): rots and rots * reflect
    def mat_mult(A, B):
        # 2x2 matrix multiply A @ B
        return ((A[0][0]*B[0][0] + A[0][1]*B[1][0],
                 A[0][0]*B[0][1] + A[0][1]*B[1][1]),
                (A[1][0]*B[0][0] + A[1][1]*B[1][0],
                 A[1][0]*B[0][1] + A[1][1]*B[1][1]))

    rots = (
        ((1,0),(0,1)),    # identity
        ((0,-1),(1,0)),   # 90 cw
        ((-1,0),(0,-1)),  # 180
        ((0,1),(-1,0)),   # 270 cw
    )
    reflect = ((1,0),(0,-1))  # reflect columns sign (flip left/right or up/down depending composition)
    R_candidates = []
    for r in rots:
        R_candidates.append(r)
    for r in rots:
        R_candidates.append(mat_mult(r, reflect))

    # apply transform R to a coordinate (r,c)
    def apply_R(R, coord):
        r, c = coord
        a, b = R[0]
        d, e = R[1]
        # new_r = a*r + b*c ; new_c = d*r + e*c
        return (a*r + b*c, d*r + e*c)

    # Work on output copy
    output = [row[:] for row in grid]

    # For each target component (not canonical) try to find mapping(s) and choose best mapping based on conflicts
    for comp in comps:
        if set(comp) == P_set:
            continue  # skip canonical itself
        Q = comp
        Q_set = set(Q)

        candidates = []
        # anchor canonical p_anchor choose first canonical coord (choose a stable anchor)
        # We'll use the canonical's top-left cell as p_anchor for consistency
        p_anchor = min(P)
        # try all R and all q_anchor
        for R in R_candidates:
            for q_anchor in Q:
                rp = apply_R(R, p_anchor)
                t_row = q_anchor[0] - rp[0]
                t_col = q_anchor[1] - rp[1]
                ok = True
                # Check that R*p + t maps exactly onto Q_set (bijection)
                for p in P:
                    mapped = apply_R(R, p)
                    mapped = (mapped[0] + t_row, mapped[1] + t_col)
                    if mapped not in Q_set:
                        ok = False
                        break
                if ok:
                    candidates.append((R, (t_row, t_col)))

        if not candidates:
            # no rigid mapping found; skip this component
            continue

        # Evaluate candidates for conflicts and placements
        best = None  # tuple (conflicts, -placements, R, t)
        for R, t in candidates:
            conflicts = 0
            placements = 0
            t_row, t_col = t
            for (decpos, color) in canonical_decorations:
                mapped = apply_R(R, decpos)
                target = (mapped[0] + t_row, mapped[1] + t_col)
                rr, cc = target
                # out of bounds is a conflict (we'll penalize heavily)
                if not (0 <= rr < rows and 0 <= cc < cols):
                    conflicts += 1000
                    continue
                existing = grid[rr][cc]
                if existing != 0 and existing != color:
                    # conflict with a different color in the original grid
                    conflicts += 1
                else:
                    # legal placement (either cell empty or same color already)
                    placements += 1
            key = (conflicts, -placements, R, t)
            if best is None or key < best:
                best = key

        if best is None:
            continue
        # apply best mapping
        _, _, R_best, t_best = best
        t_row, t_col = t_best
        for (decpos, color) in canonical_decorations:
            mapped = apply_R(R_best, decpos)
            target = (mapped[0] + t_row, mapped[1] + t_col)
            rr, cc = target
            if 0 <= rr < rows and 0 <= cc < cols:
                if output[rr][cc] == 0:
                    output[rr][cc] = color
                # otherwise, don't overwrite

    return output