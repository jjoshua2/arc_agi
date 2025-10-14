from collections import deque, Counter
from typing import List, Tuple, Set

def transform(grid_lst: List[List[int]]) -> List[List[int]]:
    if not grid_lst or not grid_lst[0]:
        return grid_lst

    h = len(grid_lst)
    w = len(grid_lst[0])
    G = grid_lst
    out = [row[:] for row in G]

    # Helpers
    def neighbors4(r, c):
        if r > 0: yield r-1, c
        if r+1 < h: yield r+1, c
        if c > 0: yield r, c-1
        if c+1 < w: yield r, c+1

    # 1) background and base colors via frequency
    counts = Counter()
    for r in range(h):
        for c in range(w):
            counts[G[r][c]] += 1
    if not counts:
        return out

    bg = counts.most_common(1)[0][0]
    # base = most frequent non-bg
    non_bg = [(col, cnt) for col, cnt in counts.items() if col != bg]
    if not non_bg:
        return out
    base = max(non_bg, key=lambda x: x[1])[0]

    # 2) Get base components and their bounding boxes
    visited = [[False]*w for _ in range(h)]
    rects: List[Tuple[int, int, int, int, Set[Tuple[int, int]]]] = []  # (minr, minc, maxr, maxc, cells set)

    for r in range(h):
        for c in range(w):
            if not visited[r][c] and G[r][c] == base:
                # BFS for a base component
                q = deque([(r, c)])
                visited[r][c] = True
                cells = [(r, c)]
                minr = maxr = r
                minc = maxc = c
                while q:
                    cr, cc = q.popleft()
                    for nr, nc in neighbors4(cr, cc):
                        if not visited[nr][nc] and G[nr][nc] == base:
                            visited[nr][nc] = True
                            q.append((nr, nc))
                            cells.append((nr, nc))
                            if nr < minr: minr = nr
                            if nr > maxr: maxr = nr
                            if nc < minc: minc = nc
                            if nc > maxc: maxc = nc
                rects.append((minr, minc, maxr, maxc, set(cells)))

    # function: is a cell inside any base rectangle bounding box?
    def inside_any_rect(r, c) -> bool:
        for (minr, minc, maxr, maxc, _cells) in rects:
            if minr <= r <= maxr and minc <= c <= maxc:
                return True
        return False

    # 3) Markers: non-base cells inside rectangles
    marker_positions: List[Tuple[int, int]] = []
    for (minr, minc, maxr, maxc, _cells) in rects:
        for rr in range(minr, maxr+1):
            for cc in range(minc, maxc+1):
                if G[rr][cc] != base:
                    marker_positions.append((rr, cc))

    if not marker_positions:
        # No markers: nothing to draw
        return out

    # marker color = most frequent color among markers
    marker_counts = Counter(G[r][c] for r, c in marker_positions)
    marker_color = marker_counts.most_common(1)[0][0]

    # 4) Legend cluster: cells not in {bg, base} and located outside all rectangles
    legend_candidates = [[False]*w for _ in range(h)]
    all_non_base_non_bg_colors = set()
    for r in range(h):
        for c in range(w):
            if G[r][c] != bg and G[r][c] != base:
                all_non_base_non_bg_colors.add(G[r][c])
                if not inside_any_rect(r, c):
                    legend_candidates[r][c] = True

    # find connected components on legend_candidates (4-conn), choose largest
    comp_id = [[-1]*w for _ in range(h)]
    comps: List[List[Tuple[int, int]]] = []
    cid = 0
    for r in range(h):
        for c in range(w):
            if legend_candidates[r][c] and comp_id[r][c] == -1:
                q = deque([(r, c)])
                comp_id[r][c] = cid
                cells = [(r, c)]
                while q:
                    cr, cc = q.popleft()
                    for nr, nc in neighbors4(cr, cc):
                        if legend_candidates[nr][nc] and comp_id[nr][nc] == -1:
                            comp_id[nr][nc] = cid
                            q.append((nr, nc))
                            cells.append((nr, nc))
                comps.append(cells)
                cid += 1

    if not comps:
        # No legend outside rectangles: fall back to a simple rule:
        # draw ribbons of the non-marker color (if any) and keep marker cell unchanged
        # (This is a safe fallback; should not happen in provided tasks.)
        palette = list(all_non_base_non_bg_colors)
        other_colors = [col for col in palette if col != marker_color]
        s = other_colors[0] if other_colors else marker_color
        c0 = marker_color
        a = None
    else:
        # Choose largest component as legend
        legend_cells = max(comps, key=len)
        legend_set = set(legend_cells)
        # Colors present in legend
        legend_colors = set(G[r][c] for r, c in legend_cells)

        # Find center in legend: max degree in legend_set
        best_deg = -1
        center_rc = legend_cells[0]
        for r, c in legend_cells:
            deg = 0
            for nr, nc in neighbors4(r, c):
                if (nr, nc) in legend_set:
                    deg += 1
            if deg > best_deg:
                best_deg = deg
                center_rc = (r, c)

        cr0, cc0 = center_rc
        c0 = G[cr0][cc0]

        # Neighbor colors around center (if present in legend)
        tcol = G[cr0-1][cc0] if (cr0-1, cc0) in legend_set else None
        bcol = G[cr0+1][cc0] if (cr0+1, cc0) in legend_set else None
        lcol = G[cr0][cc0-1] if (cr0, cc0-1) in legend_set else None
        rcol = G[cr0][cc0+1] if (cr0, cc0+1) in legend_set else None

        # Determine ribbon color s
        # Case: 3 or more legend colors
        if len(legend_colors) >= 3:
            s = None
            # Prefer a vertical bar color (top==bottom != c0)
            if tcol is not None and bcol is not None and tcol == bcol and tcol != c0:
                s = tcol
            # Fallback: most frequent legend color excluding c0
            if s is None:
                color_counts_in_legend = Counter(G[r][c] for r, c in legend_cells)
                # avoid choosing c0
                for col, _cnt in color_counts_in_legend.most_common():
                    if col != c0:
                        s = col
                        break
            # Accent color: the remaining legend color (if any) not in {c0, s}
            rem = [col for col in legend_colors if col not in (c0, s)]
            a = rem[0] if rem else None
        else:
            # exactly 1 or 2 legend colors; practically it's 2
            others = [col for col in legend_colors if col != c0]
            if others:
                only = others[0]
                # If markers equal the legend center color, ribbons use the other color, else ribbons use c0
                s = only if marker_color == c0 else c0
                # Accent if there is a color not in {c0, s} (possible only when s==c0 causing removal set still leaving one color)
                rem = [col for col in legend_colors if col not in (c0, s)]
                a = rem[0] if rem else None
            else:
                # degenerate: legend has just the center color
                s = c0
                a = None

    # Build quick rectangle lookup list to speed "which rectangle contains (r,c)"
    # Also store marker positions per rectangle (only those with marker_color)
    rect_marker_map: List[List[Tuple[int, int]]] = []
    for (minr, minc, maxr, maxc, _cells) in rects:
        ms = []
        for rr in range(minr, maxr+1):
            for cc in range(minc, maxc+1):
                if G[rr][cc] != base and G[rr][cc] == marker_color:
                    ms.append((rr, cc))
        rect_marker_map.append(ms)

    # 5) Draw ribbons in each rectangle
    for (idx, (minr, minc, maxr, maxc, _cells)) in enumerate(rects):
        markers_here = rect_marker_map[idx]
        if not markers_here:
            continue
        for (mr, mc) in markers_here:
            # Horizontal ribbon
            for c in range(minc, maxc+1):
                out[mr][c] = s
            # Vertical ribbon
            for r in range(minr, maxr+1):
                out[r][mc] = s
            # Center cell: keep marker if it equals legend center color, otherwise leave as ribbon color
            if marker_color == c0:
                out[mr][mc] = c0
            # Accent on immediate left/right inside the rectangle
            if a is not None:
                if mc-1 >= minc:
                    out[mr][mc-1] = a
                if mc+1 <= maxc:
                    out[mr][mc+1] = a

    return out