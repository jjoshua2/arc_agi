from typing import List, Tuple

def transform(grid: List[List[int]]) -> List[List[int]]:
    if not grid or not grid[0]:
        return [row[:] for row in grid]
    rows, cols = len(grid), len(grid[0])

    # Count colors (non-zero)
    cnt = [0]*10
    for r in range(rows):
        for c in range(cols):
            v = grid[r][c]
            if 1 <= v <= 9:
                cnt[v] += 1
    # pick the most frequent non-zero color as backbone
    backbone = max(range(1,10), key=lambda x: cnt[x])

    # Get all backbone cells
    backbone_cells = set()
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == backbone:
                backbone_cells.add((r,c))

    # Helper: 4-neighbors
    def neighbors4(r:int,c:int):
        for dr, dc in ((1,0),(-1,0),(0,1),(0,-1)):
            nr, nc = r+dr, c+dc
            if 0 <= nr < rows and 0 <= nc < cols:
                yield nr, nc

    # Identify "far seeds": seeds of colors {1,2,3,4} (except the backbone color)
    # that are NOT adjacent to any backbone cell.
    seed_colors = {1,2,3,4}
    far_seeds: List[Tuple[int,int,int]] = []
    for r in range(rows):
        for c in range(cols):
            v = grid[r][c]
            if v in seed_colors and v != backbone:
                adj_backbone = any((nr, nc) in backbone_cells for nr, nc in neighbors4(r,c))
                if not adj_backbone:
                    far_seeds.append((r,c,v))

    # Build quick lookup by color
    seeds_by_color = {1:[], 2:[], 3:[], 4:[]}
    for r,c,v in far_seeds:
        seeds_by_color[v].append((r,c))

    # Start fresh: only keep far seeds, everything else zero, then add connections
    out = [[0 for _ in range(cols)] for _ in range(rows)]
    for r,c,v in far_seeds:
        out[r][c] = v

    # Utility to draw a segment (exclusive of endpoints)
    def draw_segment(r1,c1,r2,c2):
        if r1 == r2:
            r = r1
            c_start = min(c1,c2)+1
            c_end = max(c1,c2)
            for cc in range(c_start, c_end):
                if out[r][cc] == 0:  # do not overwrite seeds
                    out[r][cc] = backbone
        elif c1 == c2:
            c = c1
            r_start = min(r1,r2)+1
            r_end = max(r1,r2)
            for rr in range(r_start, r_end):
                if out[rr][c] == 0:
                    out[rr][c] = backbone

    # For each blue seed (1), connect to a partner and then connect yellow if aligned.
    used_pairs = set()  # track (r1,c1,rp,cp) to avoid double work
    for r1, c1 in seeds_by_color[1]:
        partner = None  # (r,c,color)
        # prefer 2 if aligned
        best_dist = None
        for r2, c2 in seeds_by_color[2]:
            if r2 == r1 or c2 == c1:
                d = abs(r2-r1) + abs(c2-c1)
                if best_dist is None or d < best_dist:
                    best_dist = d
                    partner = (r2,c2,2)
        # else try a 3 if aligned
        if partner is None:
            for r3, c3 in seeds_by_color[3]:
                if r3 == r1 or c3 == c1:
                    d = abs(r3-r1) + abs(c3-c1)
                    if best_dist is None or d < best_dist:
                        best_dist = d
                        partner = (r3,c3,3)
        if partner is None:
            continue
        rP, cP, colP = partner
        key = (r1,c1,rP,cP) if (r1,c1) <= (rP,cP) else (rP,cP,r1,c1)
        if key in used_pairs:
            continue
        used_pairs.add(key)

        # Draw the straight segment between 1 and its partner
        draw_segment(r1,c1,rP,cP)

        # Connect a yellow seed (4) perpendicularly if aligned with that segment
        # If horizontal stem (same row), find yellow with same column between min/max columns
        if r1 == rP:
            row_stem = r1
            cmin, cmax = sorted([c1,cP])
            # choose any yellow whose column is within [cmin, cmax], pick closest by vertical distance
            candidate = None
            best_vdist = None
            for ry, cy in seeds_by_color[4]:
                if cmin <= cy <= cmax:
                    vdist = abs(ry - row_stem)
                    if best_vdist is None or vdist < best_vdist:
                        best_vdist = vdist
                        candidate = (ry, cy)
            if candidate is not None:
                ry, cy = candidate
                # connect vertically from yellow (ry,cy) to the stem row at same column
                draw_segment(ry, cy, row_stem, cy)
        else:
            # vertical stem (same column)
            col_stem = c1
            rmin, rmax = sorted([r1,rP])
            candidate = None
            best_hdist = None
            for ry, cy in seeds_by_color[4]:
                if rmin <= ry <= rmax:
                    hdist = abs(cy - col_stem)
                    if best_hdist is None or hdist < best_hdist:
                        best_hdist = hdist
                        candidate = (ry, cy)
            if candidate is not None:
                ry, cy = candidate
                # connect horizontally from yellow to the stem column at same row
                draw_segment(ry, cy, ry, col_stem)

    return out