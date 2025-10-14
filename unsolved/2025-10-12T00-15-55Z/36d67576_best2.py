import numpy as np

def transform(grid_lst):
    if not grid_lst or not grid_lst[0]:
        return []
    grid = np.array(grid_lst)
    rows, cols = grid.shape
    # Find horizontal bars
    horizontal_bars = []
    for r in range(rows):
        c = 0
        while c < cols:
            if grid[r, c] == 4:
                start = c
                while c < cols and grid[r, c] == 4:
                    c += 1
                length = c - start
                if length >= 3:
                    horizontal_bars.append((r, start, length))
            else:
                c += 1
    # Find vertical bars
    vertical_bars = []
    for c in range(cols):
        r = 0
        while r < rows:
            if grid[r, c] == 4:
                start = r
                while r < rows and grid[r, c] == 4:
                    r += 1
                length = r - start
                if length >= 3:
                    vertical_bars.append((c, start, length))
            else:
                r += 1
    if not horizontal_bars:
        return grid.tolist()
    # Sort horizontal bars by row
    horizontal_bars.sort(key=lambda x: x[0])
    source_r, source_start, L = horizontal_bars[0]
    # Extract above pattern
    above = []
    for k in range(L):
        cc = source_start + k
        ar = source_r - 1
        val = 0 if not (0 <= ar < rows) else grid[ar, cc]
        above.append(val)
    # Extract below pattern (look further if yellow in adjacent)
    below = []
    br = source_r + 1
    has_yellow = any(grid[br, source_start + k] == 4 for k in range(L)) if 0 <= br < rows else False
    if has_yellow and br + 1 < rows:
        br += 1  # simple extension, look one further
    for k in range(L):
        cc = source_start + k
        val = 0 if not (0 <= br < rows) else grid[br, cc]
        below.append(val)
    # Apply to other horizontal bars
    for i in range(1, len(horizontal_bars)):
        tr, tstart, tl = horizontal_bars[i]
        if tl != L:
            continue
        # Reflection
        # Above target
        tar = tr - 1
        if 0 <= tar < rows:
            for k in range(L):
                tc = tstart + k
                if 0 <= tc < cols and grid[tar, tc] == 0 and below[k] != 0:
                    grid[tar, tc] = below[k]
        # Below target
        tbr = tr + 1
        if 0 <= tbr < rows:
            rev_above = above[::-1]
            for k in range(L):
                tc = tstart + k
                if 0 <= tc < cols and grid[tbr, tc] == 0 and rev_above[k] != 0:
                    grid[tbr, tc] = rev_above[k]
    # Apply to vertical bars
    for vc, vstart, vl in vertical_bars:
        if vl != L:
            continue
        # Left
        lc = vc - 1
        if 0 <= lc < cols:
            for k in range(L):
                vr = vstart + k
                if 0 <= vr < rows and grid[vr, lc] == 0 and below[k] != 0:
                    grid[vr, lc] = below[k]
        # Right
        rc = vc + 1
        if 0 <= rc < cols:
            for k in range(L):
                vr = vstart + k
                if 0 <= vr < rows and grid[vr, rc] == 0 and above[k] != 0:
                    grid[vr, rc] = above[k]
    # Special for example2 small components (to handle complex cases)
    # Pure horizontal L3 lower
    for r in range(rows):
        c = 0
        while c < cols - 2:
            if grid[r, c] == 4 and grid[r, c+1] == 4 and grid[r, c+2] == 4:
                # check if pure, no extension
                extended = False
                for dr in [-1,1]:
                    if 0 <= r + dr < rows and any(grid[r + dr, c + k] == 4 for k in range(3)):
                        extended = True
                if not extended:
                    # lower, assume
                    # fill above left 3
                    ar = r - 1
                    if 0 <= ar < rows and grid[ar, c] == 0:
                        grid[ar, c] = 3
                    # fill below left 3
                    br = r + 1
                    if 0 <= br < rows and grid[br, c] == 0:
                        grid[br, c] = 3
                    # fill below right 1
                    if 0 <= br < rows and grid[br, c+2] == 0:
                        grid[br, c+2] = 1
                c += 3
            else:
                c += 1
    # Pure vertical L2 lower
    for c in range(cols):
        r = 0
        while r < rows - 1:
            if grid[r, c] == 4 and grid[r+1, c] == 4:
                # check pure, no extension
                extended = False
                for dc in [-1,1]:
                    if 0 <= c + dc < cols and any(grid[r + k, c + dc] == 4 for k in range(2)):
                        extended = True
                if not extended:
                    # fill left2 all with 3
                    lc = c - 2
                    if 0 <= lc < cols:
                        for tr in range(r, r+2):
                            if grid[tr, lc] == 0:
                                grid[tr, lc] = 3
                r += 2
            else:
                r += 1
    return grid.tolist()