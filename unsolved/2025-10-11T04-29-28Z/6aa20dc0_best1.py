def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    # Copy input
    if not grid_lst:
        return []
    grid = [row[:] for row in grid_lst]
    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0

    # Find background color as the most frequent color
    freq = {}
    for r in range(rows):
        for c in range(cols):
            v = grid[r][c]
            freq[v] = freq.get(v, 0) + 1
    background = max(freq.keys(), key=lambda k: freq[k])

    # Flood-fill connected components of non-background cells (4-connected)
    visited = [[False] * cols for _ in range(rows)]
    components = []  # each comp: dict with 'cells' list and 'colors' set, bounding box
    for r in range(rows):
        for c in range(cols):
            if visited[r][c]:
                continue
            if grid[r][c] == background:
                continue
            # BFS / stack
            stack = [(r, c)]
            visited[r][c] = True
            cells = []
            colors = set()
            while stack:
                x, y = stack.pop()
                cells.append((x, y))
                colors.add(grid[x][y])
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < rows and 0 <= ny < cols and not visited[nx][ny] and grid[nx][ny] != background:
                        visited[nx][ny] = True
                        stack.append((nx, ny))
            minr = min(x for x, y in cells)
            maxr = max(x for x, y in cells)
            minc = min(y for x, y in cells)
            maxc = max(y for x, y in cells)
            bbox_area = (maxr - minr + 1) * (maxc - minc + 1)
            components.append({
                'cells': cells,
                'colors': colors,
                'minr': minr, 'maxr': maxr, 'minc': minc, 'maxc': maxc,
                'area': bbox_area
            })

    # Choose motif component: one with >1 distinct colors, prefer more distinct colors then smaller area
    motif_comp = None
    best_key = None
    for comp in components:
        if len(comp['colors']) <= 1:
            continue
        key = (len(comp['colors']), -comp['area'])  # maximize distinct colors, minimize area
        if motif_comp is None or key > best_key:
            motif_comp = comp
            best_key = key

    # If no motif found, return original grid
    if motif_comp is None:
        return grid

    # Extract motif bounding box
    r0, r1 = motif_comp['minr'], motif_comp['maxr']
    c0, c1 = motif_comp['minc'], motif_comp['maxc']
    mh = r1 - r0 + 1
    mw = c1 - c0 + 1
    motif = [[grid[r0 + i][c0 + j] for j in range(mw)] for i in range(mh)]

    # Count colors inside motif
    motif_color_count = {}
    for i in range(mh):
        for j in range(mw):
            v = motif[i][j]
            motif_color_count[v] = motif_color_count.get(v, 0) + 1

    # Collect anchors based on motif cells that are unique inside motif (count == 1)
    anchors = []  # list of tuples (i0, j0, anchor_r, anchor_c)
    for i0 in range(mh):
        for j0 in range(mw):
            v = motif[i0][j0]
            # skip background and skip colors that are not unique inside motif
            if v == background:
                continue
            if motif_color_count.get(v, 0) != 1:
                continue
            # find all occurrences of v outside motif bounding box in the original grid
            for r in range(rows):
                for c in range(cols):
                    # skip positions inside original motif bounding box
                    if r0 <= r <= r1 and c0 <= c <= c1:
                        continue
                    if grid[r][c] == v:
                        anchors.append((i0, j0, r, c))

    # Prepare output as a copy of original grid
    output = [row[:] for row in grid]

    # Paste motif for each anchor (use original motif values)
    for (i0, j0, ar, ac) in anchors:
        top = ar - i0
        left = ac - j0
        # ensure placement fits inside grid
        if top < 0 or left < 0 or top + mh > rows or left + mw > cols:
            continue
        for i in range(mh):
            for j in range(mw):
                output[top + i][left + j] = motif[i][j]

    return output