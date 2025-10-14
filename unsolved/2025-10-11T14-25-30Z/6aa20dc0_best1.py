from collections import Counter, deque

def transform(grid: list[list[int]]) -> list[list[int]]:
    if not grid or not grid[0]:
        return []
    h = len(grid)
    w = len(grid[0])

    # 1) background color = most frequent color
    cnt = Counter()
    for r in range(h):
        for c in range(w):
            cnt[grid[r][c]] += 1
    bg_color = cnt.most_common(1)[0][0]

    # 2) locate the 3x3 seed window:
    seed_candidates = []
    for r in range(h - 2):
        for c in range(w - 2):
            cells = [grid[r + dr][c + dc] for dr in range(3) for dc in range(3)]
            non_bg = [v for v in cells if v != bg_color]
            non_bg_set = set(non_bg)
            # candidate if there's more than one distinct non-background color (multi-colored)
            if len(non_bg_set) > 1:
                # metric: (distinct non-bg colors, number of non-bg cells)
                seed_candidates.append(((len(non_bg_set), len(non_bg)), r, c))
    if not seed_candidates:
        # nothing to do (no multi-colored 3x3 found)
        return [row[:] for row in grid]

    # pick best candidate by metrics (most distinct non-bg colors, then most non-bg cells)
    seed_candidates.sort(reverse=True)
    _, seed_r, seed_c = seed_candidates[0]
    # extract seed 3x3
    seed = [[grid[seed_r + i][seed_c + j] for j in range(3)] for i in range(3)]

    # 3) find connected components (4-connected) of non-background cells excluding seed area
    visited = [[False] * w for _ in range(h)]
    def in_seed_area(rr, cc):
        return seed_r <= rr <= seed_r + 2 and seed_c <= cc <= seed_c + 2

    squares = []  # list of dicts {color, minr, minc, maxr, maxc, size}
    for r in range(h):
        for c in range(w):
            if grid[r][c] == bg_color:
                continue
            if in_seed_area(r, c):
                continue
            if visited[r][c]:
                continue
            color = grid[r][c]
            # BFS to gather component
            q = deque()
            q.append((r, c))
            visited[r][c] = True
            comp = []
            while q:
                pr, pc = q.popleft()
                comp.append((pr, pc))
                for nr, nc in ((pr-1, pc), (pr+1, pc), (pr, pc-1), (pr, pc+1)):
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr][nc] and grid[nr][nc] == color:
                        visited[nr][nc] = True
                        q.append((nr, nc))
            # bounding box
            minr = min(x[0] for x in comp)
            maxr = max(x[0] for x in comp)
            minc = min(x[1] for x in comp)
            maxc = max(x[1] for x in comp)
            height = maxr - minr + 1
            width = maxc - minc + 1
            if height == width and len(comp) == height * width:
                squares.append({
                    'color': color,
                    'minr': minr,
                    'minc': minc,
                    'maxr': maxr,
                    'maxc': maxc,
                    'size': height
                })
    if not squares:
        # nothing to tile
        return [row[:] for row in grid]

    # 4) determine k as the most common square size among squares
    size_counts = Counter(sq['size'] for sq in squares)
    k = size_counts.most_common(1)[0][0]

    # 5) bounding rectangle containing all squares
    minr = min(sq['minr'] for sq in squares)
    minc = min(sq['minc'] for sq in squares)
    maxr = max(sq['maxr'] for sq in squares)
    maxc = max(sq['maxc'] for sq in squares)

    # 6) generate transforms of seed: rotations and horizontal flip variants
    def rotate90(mat):
        # rotate 90 degrees clockwise
        return [[mat[2 - j][i] for j in range(3)] for i in range(3)]
    def flip_h(mat):
        return [row[::-1] for row in mat]

    mats = []
    cur = seed
    for _ in range(4):
        mats.append(cur)
        mats.append(flip_h(cur))
        cur = rotate90(cur)
    # deduplicate while preserving order
    transforms = []
    for m in mats:
        if all(m != ex for ex in transforms):
            transforms.append(m)

    # 7) choose transform that matches the colors of the uniform blocks at their top-lefts
    chosen = None
    for T in transforms:
        ok = True
        for sq in squares:
            rtop = sq['minr']
            ctop = sq['minc']
            # compute which macro-cell (0..2, 0..2) this top-left falls into
            macro_r = ((rtop - minr) // k) % 3
            macro_c = ((ctop - minc) // k) % 3
            if T[macro_r][macro_c] != sq['color']:
                ok = False
                break
        if ok:
            chosen = T
            break
    if chosen is None:
        # fallback: use original seed (no transform matches perfectly)
        chosen = transforms[0]

    # 8) overlay tiled/scaled chosen across bounding rectangle
    output = [row[:] for row in grid]
    for r in range(minr, maxr + 1):
        for c in range(minc, maxc + 1):
            macro_r = ((r - minr) // k) % 3
            macro_c = ((c - minc) // k) % 3
            output[r][c] = chosen[macro_r][macro_c]

    return output