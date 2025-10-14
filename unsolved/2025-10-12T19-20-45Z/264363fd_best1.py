from collections import deque
from typing import List, Tuple, Dict

def transform(input_grid: List[List[int]]) -> List[List[int]]:
    if not input_grid or not input_grid[0]:
        return []
    rows, cols = len(input_grid), len(input_grid[0])
    grid = [row[:] for row in input_grid]

    # Helpers
    def in_bounds(r: int, c: int) -> bool:
        return 0 <= r < rows and 0 <= c < cols

    # 1) Background color = most frequent color
    from collections import Counter
    cnt = Counter()
    for r in range(rows):
        for c in range(cols):
            cnt[grid[r][c]] += 1
    background = max(cnt.items(), key=lambda x: x[1])[0]

    # 2) Find connected components by color (excluding background)
    visited = [[False]*cols for _ in range(rows)]
    components = []  # list of (color, cells)
    dirs4 = [(1,0),(-1,0),(0,1),(0,-1)]
    for r in range(rows):
        for c in range(cols):
            if not visited[r][c] and grid[r][c] != background:
                col = grid[r][c]
                q = deque([(r,c)])
                visited[r][c] = True
                cells = []
                while q:
                    x,y = q.popleft()
                    cells.append((x,y))
                    for dr,dc in dirs4:
                        nx,ny = x+dr, y+dc
                        if in_bounds(nx,ny) and not visited[nx][ny] and grid[nx][ny] == col:
                            visited[nx][ny] = True
                            q.append((nx,ny))
                components.append((col, cells))

    # 3) Identify box components: high rectangular fill-ratio and decent size
    def bbox(cells):
        rs = [r for r,_ in cells]
        cs = [c for _,c in cells]
        return min(rs), min(cs), max(rs), max(cs)

    boxes = []  # list of dicts with keys: color, r0,c0,r1,c1
    for col, cells in components:
        if len(cells) < 8:
            continue
        r0,c0,r1,c1 = bbox(cells)
        area = (r1-r0+1)*(c1-c0+1)
        fill_ratio = len(cells)/area
        # Good rectangle if axis-aligned, largely filled, and spans at least 3x3
        if (r1-r0+1) >= 3 and (c1-c0+1) >= 3 and fill_ratio >= 0.85:
            boxes.append({'color': col, 'r0': r0, 'c0': c0, 'r1': r1, 'c1': c1})

    if not boxes:
        return grid

    # 4) Determine marker color C: most common non-box-color inside boxes
    inside_counts = Counter()
    for box in boxes:
        B = box['color']
        for r in range(box['r0'], box['r1']+1):
            for c in range(box['c0'], box['c1']+1):
                v = grid[r][c]
                if v != B:
                    inside_counts[v] += 1
    if not inside_counts:
        return grid
    marker_color = inside_counts.most_common(1)[0][0]

    # Helper: check if a cell is inside any box
    def in_any_box(r: int, c: int) -> bool:
        for b in boxes:
            if b['r0'] <= r <= b['r1'] and b['c0'] <= c <= b['c1']:
                return True
        return False

    # 5) Find legend center: a marker_color outside boxes with informative neighbors
    legend_centers = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == marker_color and not in_any_box(r,c):
                # score: number of neighboring non-background and not marker colors
                sc = 0
                for dr,dc in dirs4:
                    nr,nc = r+dr, c+dc
                    if in_bounds(nr,nc):
                        v = grid[nr][nc]
                        if v != background and v != marker_color:
                            sc += 1
                legend_centers.append((sc, r, c))
    if not legend_centers:
        # If no legend center outside, nothing to do
        return grid
    legend_centers.sort(reverse=True)
    _, cr, cc = legend_centers[0]  # pick the one with highest score

    # 6) Determine ribbon colors and directions from legend 4-neighborhood
    up = grid[cr-1][cc] if cr-1 >= 0 else None
    down = grid[cr+1][cc] if cr+1 < rows else None
    left = grid[cr][cc-1] if cc-1 >= 0 else None
    right = grid[cr][cc+1] if cc+1 < cols else None

    vertical_color = None
    if up is not None and down is not None and up == down and up != background and up != marker_color:
        vertical_color = up
    horizontal_color = None
    if left is not None and right is not None and left == right and left != background and left != marker_color:
        horizontal_color = left

    # Decide which ribbons to draw
    draw_vertical = vertical_color is not None
    draw_horizontal = horizontal_color is not None
    # If both exist but differ, only draw vertical (as in example 3)
    if draw_vertical and draw_horizontal and vertical_color != horizontal_color:
        draw_horizontal = False

    # 7) Collect accent offsets (3x3 around legend center) that are not ribbon colors nor background
    accent_offsets: Dict[Tuple[int,int], int] = {}
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            rr, cc2 = cr+dr, cc+dc
            if in_bounds(rr, cc2):
                col = grid[rr][cc2]
                # skip background and any ribbon colors used
                if col == background:
                    continue
                if draw_vertical and col == vertical_color:
                    continue
                if draw_horizontal and col == horizontal_color:
                    continue
                # record accent color at this relative offset
                accent_offsets[(dr, dc)] = col

    # 8) Apply ribbons to each marker inside each box
    out = [row[:] for row in grid]

    for box in boxes:
        r0, c0, r1, c1 = box['r0'], box['c0'], box['r1'], box['c1']
        B = box['color']
        # Find all marker positions in this box
        markers = []
        for r in range(r0, r1+1):
            for c in range(c0, c1+1):
                if grid[r][c] == marker_color:
                    markers.append((r,c))
        # Draw for each marker
        for mr, mc in markers:
            if draw_vertical:
                for r in range(r0, r1+1):
                    out[r][mc] = vertical_color
            if draw_horizontal:
                for c in range(c0, c1+1):
                    out[mr][c] = horizontal_color
            # Place center back to marker color
            out[mr][mc] = marker_color
            # Place accents
            for (dr, dc), acol in accent_offsets.items():
                rr, cc2 = mr + dr, mc + dc
                if r0 <= rr <= r1 and c0 <= cc2 <= c1:
                    out[rr][cc2] = acol

    return out