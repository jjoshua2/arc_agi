from collections import deque, defaultdict

def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    if not grid_lst or not grid_lst[0]:
        return grid_lst

    rows = len(grid_lst)
    cols = len(grid_lst[0])

    # Work on a copy for output
    out = [row[:] for row in grid_lst]

    # Determine background color (corner)
    background = grid_lst[0][0]

    # Count colors to pick rectangle color (most common excluding background)
    counts = defaultdict(int)
    for r in range(rows):
        for c in range(cols):
            counts[grid_lst[r][c]] += 1

    rect_color = None
    max_count = -1
    for color, cnt in counts.items():
        if color == background:
            continue
        if cnt > max_count:
            max_count = cnt
            rect_color = color

    if rect_color is None:
        return out

    # Find connected components of rect_color
    visited = [[False]*cols for _ in range(rows)]
    dirs = [(-1,0),(1,0),(0,-1),(0,1)]
    components = []  # list of lists of (r,c)

    for r in range(rows):
        for c in range(cols):
            if not visited[r][c] and grid_lst[r][c] == rect_color:
                q = deque()
                q.append((r,c))
                visited[r][c] = True
                comp = []
                while q:
                    x,y = q.popleft()
                    comp.append((x,y))
                    for dx,dy in dirs:
                        nx, ny = x+dx, y+dy
                        if 0 <= nx < rows and 0 <= ny < cols and (not visited[nx][ny]) and grid_lst[nx][ny] == rect_color:
                            visited[nx][ny] = True
                            q.append((nx, ny))
                components.append(comp)

    # For each component, compute bounding box and, if big enough, draw cross of purple(8)
    for comp in components:
        minr = min(r for r,c in comp)
        maxr = max(r for r,c in comp)
        minc = min(c for r,c in comp)
        maxc = max(c for r,c in comp)

        # Require a bounding box at least 3x3 (so it has a clear center row/col)
        if (maxr - minr) >= 2 and (maxc - minc) >= 2:
            rc = (minr + maxr) // 2
            cc = (minc + maxc) // 2

            comp_set = set(comp)

            # Horizontal center row: set rect_color cells in component to 8
            for ccx in range(minc, maxc + 1):
                if (rc, ccx) in comp_set and grid_lst[rc][ccx] == rect_color:
                    out[rc][ccx] = 8

            # Vertical center column: set rect_color cells in component to 8
            for rrr in range(minr, maxr + 1):
                if (rrr, cc) in comp_set and grid_lst[rrr][cc] == rect_color:
                    out[rrr][cc] = 8

    return out