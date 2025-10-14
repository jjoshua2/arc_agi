from collections import deque, defaultdict
from copy import deepcopy

def transform(grid: list[list[int]]) -> list[list[int]]:
    if not grid or not grid[0]:
        return grid
    h = len(grid)
    w = len(grid[0])

    def in_bounds(r, c):
        return 0 <= r < h and 0 <= c < w

    # 4-neighbour deltas
    neigh4 = [(-1,0),(1,0),(0,-1),(0,1)]

    # Count occurrences of colors (exclude 0)
    counts = defaultdict(int)
    positions_by_color = defaultdict(list)
    for r in range(h):
        for c in range(w):
            v = grid[r][c]
            if v != 0:
                counts[v] += 1
                positions_by_color[v].append((r,c))

    if not counts:
        return deepcopy(grid)

    # Choose fill color F as the most frequent non-zero color
    F = max(counts.items(), key=lambda x: (x[1], x[0]))[0]

    # Build connected components of color F (4-neighbour)
    seen = [[False]*w for _ in range(h)]
    components = []
    for (r,c) in positions_by_color.get(F, []):
        if seen[r][c]:
            continue
        # BFS
        comp = []
        dq = deque()
        dq.append((r,c))
        seen[r][c] = True
        while dq:
            cr, cc = dq.popleft()
            comp.append((cr,cc))
            for dr,dc in neigh4:
                nr, nc = cr+dr, cc+dc
                if in_bounds(nr,nc) and (not seen[nr][nc]) and grid[nr][nc] == F:
                    seen[nr][nc] = True
                    dq.append((nr,nc))
        components.append(set(comp))

    # Work on a copy of grid for output; we'll update it progressively
    out = deepcopy(grid)

    # Helper rotations: cw and ccw on (dr,dc)
    def rotate_cw(dr, dc):
        # (dr,dc) -> (dc,-dr)
        return dc, -dr
    def rotate_ccw(dr, dc):
        # (dr,dc) -> (-dc, dr)
        return -dc, dr

    # Precompute adjacency of any cell to F in the current 'out' grid will be recomputed per component attempt
    def adjacent_to_F_positions(grid_state):
        adj = set()
        for r in range(h):
            for c in range(w):
                if grid_state[r][c] == F:
                    for dr,dc in neigh4:
                        nr, nc = r+dr, c+dc
                        if in_bounds(nr,nc) and grid_state[nr][nc] != 0 and grid_state[nr][nc] != F:
                            adj.add((nr,nc))
        return adj

    # For each component, attempt to find a matching orphan anchor arrangement and move it
    # We'll operate on 'out' progressively; neighbor detection for the component uses original positions of that comp
    original_grid = grid  # keep original for comp geometry reference

    # Precompute for speed: for each non-F cell whether it's adjacent to ANY F in the original grid.
    # (But we will recompute orphans dynamically based on 'out' when trying to map.)
    # We'll use original component geometry (comp cells and their adjacent neighbor anchors).
    for comp in components:
        # Collect neighbor anchors (positions with color != 0 and != F adjacent to any cell in comp)
        neighbor_positions = {}  # (r,c) -> color
        for (fr,fc) in comp:
            for dr,dc in neigh4:
                nr, nc = fr+dr, fc+dc
                if in_bounds(nr,nc):
                    val = original_grid[nr][nc]
                    if val != 0 and val != F:
                        neighbor_positions[(nr,nc)] = val
        if not neighbor_positions:
            # nothing attached; skip
            continue

        # Build list of neighbor entries
        neighbor_items = [ (pos[0], pos[1], color) for pos, color in neighbor_positions.items() ]

        # Build orphan positions by color dynamically from current out grid state
        # (An orphan is a colored cell (val != 0 and != F) that has no F neighbor in current out)
        def compute_orphans(state_grid):
            # find adjacency to F in the current state
            attached = set()
            for r in range(h):
                for c in range(w):
                    if state_grid[r][c] == F:
                        for dr,dc in neigh4:
                            nr, nc = r+dr, c+dc
                            if in_bounds(nr,nc) and state_grid[nr][nc] != 0 and state_grid[nr][nc] != F:
                                attached.add((nr,nc))
            orphans = defaultdict(list)
            for r in range(h):
                for c in range(w):
                    val = state_grid[r][c]
                    if val != 0 and val != F:
                        if (r,c) not in attached:
                            orphans[val].append((r,c))
            return orphans

        orphans = compute_orphans(out)

        # Check each neighbor color has at least one orphan candidate; if not, cannot map this component now
        neighbor_colors = set(color for (_,_,color) in neighbor_items)
        can_attempt = True
        for color in neighbor_colors:
            if color not in orphans or len(orphans[color]) == 0:
                can_attempt = False
                break
        if not can_attempt:
            # No mapping possible for this component in current out state
            continue

        # Attempt to find a rotation and pivot mapping that maps all neighbor positions to orphan positions
        mapping_found = False
        # For efficient matching: for each color, build a set for membership test
        orphan_sets = { col: set(orphans[col]) for col in orphans }

        # Try possible pivots: each neighbor position can be pivot, and each orphan of that pivot color can be pivot target
        for (pr, pc, pcol) in neighbor_items:
            if mapping_found:
                break
            target_candidates = orphan_sets.get(pcol, [])
            if not target_candidates:
                continue
            for (trp, tcp) in target_candidates:
                if mapping_found:
                    break
                # Test both rotations
                for rot in ('cw','ccw'):
                    valid = True
                    # For each neighbor, check where it would map and ensure that target equals same color in current out
                    for (nr, nc, ncol) in neighbor_items:
                        dr = nr - pr
                        dc = nc - pc
                        if rot == 'cw':
                            drp, dcp = rotate_cw(dr, dc)
                        else:
                            drp, dcp = rotate_ccw(dr, dc)
                        rr = trp + drp
                        cc = tcp + dcp
                        if not in_bounds(rr, cc):
                            valid = False
                            break
                        if out[rr][cc] != ncol:
                            # the rotated neighbor must land onto a same-colored orphan cell
                            valid = False
                            break
                    if not valid:
                        continue

                    # Mapping is valid: apply it - move the entire component comp (F cells)
                    # Use pivot (pr,pc) in component and pivot target (trp,tcp)
                    # Place F at rotated locations if target cell is 0 (do not overwrite)
                    for (fr, fc) in comp:
                        dr = fr - pr
                        dc = fc - pc
                        if rot == 'cw':
                            drp, dcp = rotate_cw(dr, dc)
                        else:
                            drp, dcp = rotate_ccw(dr, dc)
                        rr = trp + drp
                        cc = tcp + dcp
                        if in_bounds(rr, cc) and out[rr][cc] == 0:
                            out[rr][cc] = F
                    # After placement, clear original component cells (set to 0) and clear original neighbor anchors
                    for (fr, fc) in comp:
                        if in_bounds(fr, fc):
                            out[fr][fc] = 0
                    for (nr, nc, ncol) in neighbor_items:
                        if in_bounds(nr, nc):
                            out[nr][nc] = 0
                    mapping_found = True
                    break  # exit rotation loop
        # end pivot loops; if mapping_found false, no mapping for this component (skip it)
    # end for components

    return out