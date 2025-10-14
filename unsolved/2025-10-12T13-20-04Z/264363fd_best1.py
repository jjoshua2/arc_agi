from collections import deque, Counter
from typing import List, Tuple

def transform(grid: List[List[int]]) -> List[List[int]]:
    if not grid or not grid[0]:
        return [row[:] for row in grid]
    h, w = len(grid), len(grid[0])

    # Count colors to find background
    cnt = Counter()
    for r in range(h):
        cnt.update(grid[r])
    bg = cnt.most_common(1)[0][0]

    # 4-neighborhood
    dirs = [(-1,0),(1,0),(0,-1),(0,1)]

    # Connected components by color
    visited = [[False]*w for _ in range(h)]
    comps = []  # list of dicts: {color, cells, bbox}
    for r in range(h):
        for c in range(w):
            if visited[r][c]:
                continue
            color = grid[r][c]
            # we consider all components; we'll filter later
            q = deque()
            q.append((r,c))
            visited[r][c] = True
            cells = []
            minr=minc=10**9
            maxr=maxc=-1
            while q:
                x,y = q.popleft()
                cells.append((x,y))
                if x<minr: minr=x
                if x>maxr: maxr=x
                if y<minc: minc=y
                if y>maxc: maxc=y
                for dr,dc in dirs:
                    nx,ny = x+dr, y+dc
                    if 0<=nx<h and 0<=ny<w and not visited[nx][ny] and grid[nx][ny]==color:
                        visited[nx][ny]=True
                        q.append((nx,ny))
            comps.append({
                "color": color,
                "cells": cells,
                "bbox": (minr,minc,maxr,maxc),
                "size": len(cells)
            })

    # Determine box color as the largest non-background component color
    non_bg = [comp for comp in comps if comp["color"] != bg]
    if not non_bg:
        return [row[:] for row in grid]
    largest_comp = max(non_bg, key=lambda x: x["size"])
    box_color = largest_comp["color"]

    # Collect all box components (same color as largest)
    box_comps = [comp for comp in comps if comp["color"] == box_color]

    # Helper: is a position inside any box bbox?
    box_bboxes = [comp["bbox"] for comp in box_comps]
    def in_any_box(r:int,c:int)->bool:
        for (r1,c1,r2,c2) in box_bboxes:
            if r1 <= r <= r2 and c1 <= c <= c2:
                return True
        return False

    # Find marker color(s) inside boxes (any color not bg and not box_color)
    marker_counts = Counter()
    marker_positions_by_box: List[List[Tuple[int,int]]] = []
    for (r1,c1,r2,c2) in box_bboxes:
        markers_here = []
        for rr in range(r1, r2+1):
            for cc in range(c1, c2+1):
                col = grid[rr][cc]
                if col != box_color and col != bg:
                    marker_counts[col] += 1
                    markers_here.append((rr,cc))
        marker_positions_by_box.append(markers_here)

    if not marker_counts:
        # No markers inside boxes; nothing to do
        return [row[:] for row in grid]

    marker_color = marker_counts.most_common(1)[0][0]

    # Find a template marker (of marker_color) outside boxes
    templ_pos = None
    for comp in comps:
        if comp["color"] != marker_color:
            continue
        # choose any cell outside all boxes
        for (rr,cc) in comp["cells"]:
            if not in_any_box(rr,cc):
                templ_pos = (rr,cc)
                break
        if templ_pos is not None:
            break

    # Infer ribbon colors from template neighborhood
    vert_color = None
    horiz_color = None
    if templ_pos is not None:
        tr, tc = templ_pos
        # up/down same, not background, not marker_color
        upc = grid[tr-1][tc] if tr-1 >= 0 else None
        dnc = grid[tr+1][tc] if tr+1 < h else None
        if upc is not None and dnc is not None and upc == dnc and upc != bg and upc != marker_color:
            vert_color = upc
        # left/right same, not background, not marker_color
        lfc = grid[tr][tc-1] if tc-1 >= 0 else None
        rtc = grid[tr][tc+1] if tc+1 < w else None
        if lfc is not None and rtc is not None and lfc == rtc and lfc != bg and lfc != marker_color:
            horiz_color = lfc

    # If still missing a ribbon color, try to pick any other frequent color outside boxes (besides bg, box, marker)
    if vert_color is None and horiz_color is None:
        outside_counts = Counter()
        for comp in comps:
            if comp["color"] in (bg, box_color, marker_color):
                continue
            # count only cells outside boxes
            for (rr,cc) in comp["cells"]:
                if not in_any_box(rr,cc):
                    outside_counts[comp["color"]] += 1
        if outside_counts:
            # prefer the most frequent as vertical
            vert_color = outside_counts.most_common(1)[0][0]

    # Prepare output
    out = [row[:] for row in grid]

    # For each box, collect its markers (positions with marker_color), then draw ribbons
    for bbox, markers in zip(box_bboxes, marker_positions_by_box):
        r1,c1,r2,c2 = bbox
        # Collect marker positions of the chosen marker_color
        mpos = [(r,c) for (r,c) in markers if grid[r][c] == marker_color]
        if not mpos:
            continue
        # Draw per marker
        for (mr, mc) in mpos:
            # vertical ribbon
            if vert_color is not None:
                for rr in range(r1, r2+1):
                    if rr == mr and grid[rr][mc] == marker_color:
                        continue  # keep the marker color at the crossing
                    if out[rr][mc] == box_color:
                        out[rr][mc] = vert_color
            # horizontal ribbon
            if horiz_color is not None:
                for cc in range(c1, c2+1):
                    if cc == mc and grid[mr][cc] == marker_color:
                        continue  # keep the marker color at the crossing
                    if out[mr][cc] == box_color:
                        out[mr][cc] = horiz_color

    return out