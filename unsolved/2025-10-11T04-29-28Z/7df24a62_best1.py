def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    # Work on a copy
    if not grid_lst or not grid_lst[0]:
        return []
    grid_in = [row[:] for row in grid_lst]
    h = len(grid_in)
    w = len(grid_in[0])

    # Helper: find connected components of color 1 (4-connected)
    visited = [[False] * w for _ in range(h)]
    comps = []  # each comp: list of (r,c)
    directions = [(-1,0),(1,0),(0,-1),(0,1)]

    for i in range(h):
        for j in range(w):
            if grid_in[i][j] == 1 and not visited[i][j]:
                stack = [(i,j)]
                visited[i][j] = True
                cells = []
                minr = i; maxr = i; minc = j; maxc = j
                while stack:
                    r,c = stack.pop()
                    cells.append((r,c))
                    if r < minr: minr = r
                    if r > maxr: maxr = r
                    if c < minc: minc = c
                    if c > maxc: maxc = c
                    for dr,dc in directions:
                        nr,nc = r+dr, c+dc
                        if 0 <= nr < h and 0 <= nc < w and not visited[nr][nc] and grid_in[nr][nc] == 1:
                            visited[nr][nc] = True
                            stack.append((nr,nc))
                comps.append((cells, minr, minc, maxr, maxc))

    # If there are no 1-components, nothing to replicate
    if not comps:
        return [row[:] for row in grid_in]

    # For each component, compute how many 4s are inside its bounding box
    best_comp = None
    best_4count = -1
    for comp in comps:
        _, minr, minc, maxr, maxc = comp
        cnt4 = 0
        for rr in range(minr, maxr+1):
            for cc in range(minc, maxc+1):
                if grid_in[rr][cc] == 4:
                    cnt4 += 1
        # choose the component with the largest number of 4s in its bounding box
        if cnt4 > best_4count:
            best_4count = cnt4
            best_comp = comp

    # If no component had any 4 inside bounding, fallback: choose the largest 1-component
    if best_4count == 0:
        best_comp = max(comps, key=lambda c: len(c[0]))
        # recompute its bounding and 4count as best_4count
        _, minr, minc, maxr, maxc = best_comp
        best_4count = sum(1 for rr in range(minr, maxr+1) for cc in range(minc, maxc+1) if grid_in[rr][cc] == 4)

    # Extract motif bounding box and offsets
    cells, minr, minc, maxr, maxc = best_comp
    motif_h = maxr - minr + 1
    motif_w = maxc - minc + 1

    # Build set of offsets for where motif has 1s and where motif has 4s
    motif1_offsets = set()
    motif4_offsets = set()
    for r in range(minr, maxr+1):
        for c in range(minc, maxc+1):
            val = grid_in[r][c]
            dr = r - minr
            dc = c - minc
            if val == 1:
                motif1_offsets.add((dr, dc))
            elif val == 4:
                motif4_offsets.add((dr, dc))

    # If motif area doesn't fit somewhere else or motif has no width/height, return original
    if motif_h <= 0 or motif_w <= 0 or motif_h > h or motif_w > w:
        return [row[:] for row in grid_in]

    # Slide window across grid and count how many 4s are inside each h-by-w window
    window_counts = {}
    max_count = -1
    for r0 in range(0, h - motif_h + 1):
        for c0 in range(0, w - motif_w + 1):
            # skip the original motif window
            if r0 == minr and c0 == minc:
                continue
            cnt = 0
            for dr in range(motif_h):
                row = r0 + dr
                # small micro-optimization: loop columns inner
                for dc in range(motif_w):
                    col = c0 + dc
                    if grid_in[row][col] == 4:
                        cnt += 1
            window_counts[(r0,c0)] = cnt
            if cnt > max_count:
                max_count = cnt

    # If there is no window with any 4s (max_count <= 0), nothing to add
    if max_count <= 0:
        return [row[:] for row in grid_in]

    # Prepare output grid copy
    out = [row[:] for row in grid_in]

    # For all windows that achieve max_count, overlay motif1 offsets (but keep existing 4s)
    for (r0,c0), cnt in window_counts.items():
        if cnt == max_count:
            for (dr,dc) in motif1_offsets:
                rr = r0 + dr
                cc = c0 + dc
                # don't overwrite 4s
                if out[rr][cc] == 4:
                    continue
                # set motif 1
                out[rr][cc] = 1

    return out