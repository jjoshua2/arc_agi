def transform(grid: list[list[int]]) -> list[list[int]]:
    if not grid or not grid[0]:
        return grid
    h = len(grid)
    w = len(grid[0])
    if h == 0 or w == 0:
        return [row[:] for row in grid]

    # Find bg color: most frequent
    from collections import Counter
    flat = [cell for row in grid for cell in row]
    count = Counter(flat)
    bg = count.most_common(1)[0][0]

    output = [row[:] for row in grid]

    # Collect segments and blockers
    segments = []
    blockers = set()

    # Type 0: / diff = r - c
    min_d = -(w - 1)
    max_d = h - 1
    for d in range(min_d, max_d + 1):
        positions = [(r, r - d) for r in range(h) if 0 <= r - d < w]
        if len(positions) < 2:
            continue
        color_pos = {}
        for r, c in positions:
            colr = grid[r][c]
            if colr != bg:
                color_pos.setdefault(colr, []).append(r)
        for C, rs in color_pos.items():
            if len(rs) >= 2:
                min_r = min(rs)
                max_r = max(rs)
                span = max_r - min_r
                segments.append((span, C, 0, d, min_r, max_r))
                # Collect blockers based on input
                for rr in range(min_r, max_r + 1):
                    cc = rr - d
                    if grid[rr][cc] != bg and grid[rr][cc] != C:
                        blockers.add((rr, cc, 0))

    # Type 1: \ sum = r + c
    min_s = 0
    max_s = h + w - 2
    for s in range(min_s, max_s + 1):
        positions = []
        start_r = max(0, s - w + 1)
        end_r = min(h, s + 1)
        for r in range(start_r, end_r):
            c = s - r
            positions.append((r, c))
        if len(positions) < 2:
            continue
        color_pos = {}
        for r, c in positions:
            colr = grid[r][c]
            if colr != bg:
                color_pos.setdefault(colr, []).append(r)
        for C, rs in color_pos.items():
            if len(rs) >= 2:
                min_r = min(rs)
                max_r = max(rs)
                span = max_r - min_r
                segments.append((span, C, 1, s, min_r, max_r))
                # Collect blockers
                for rr in range(min_r, max_r + 1):
                    cc = s - rr
                    if 0 <= cc < w and grid[rr][cc] != bg and grid[rr][cc] != C:
                        blockers.add((rr, cc, 1))

    # Sort segments by descending span
    segments.sort(key=lambda x: -x[0])

    # Perform main fillings in order
    for span, C, typ, diagid, min_r, max_r in segments:
        if typ == 0:
            for r in range(min_r, max_r + 1):
                c = r - diagid
                if 0 <= c < w and output[r][c] == bg:
                    output[r][c] = C
        else:
            for r in range(min_r, max_r + 1):
                c = diagid - r
                if 0 <= c < w and output[r][c] == bg:
                    output[r][c] = C

    # Now propagations
    for br, bc, bt in blockers:
        E = grid[br][bc]
        pt = 1 - bt
        if pt == 0:
            # Fill entire / d = br - bc
            d = br - bc
            for r in range(h):
                c = r - d
                if 0 <= c < w and output[r][c] == bg:
                    output[r][c] = E
        else:
            # Fill entire \ s = br + bc
            s = br + bc
            start_r = max(0, s - w + 1)
            end_r = min(h, s + 1)
            for r in range(start_r, end_r):
                c = s - r
                if output[r][c] == bg:
                    output[r][c] = E

    return output