def transform(grid: list[list[int]]) -> list[list[int]]:
    if not grid or not grid[0]:
        return []
    h = len(grid)
    w = len(grid[0])
    output = [row[:] for row in grid]
    ones = [(i, j) for i in range(h) for j in range(w) if output[i][j] == 1]

    to_move = {}  # (r, c): list of original positions moving here

    for r, c in ones:
        # Horizontal targets in row r
        targets_h = []
        i = 0
        while i < w:
            if output[r][i] == 2:
                l = i
                while i < w and output[r][i] == 2:
                    i += 1
                rr = i - 1
                # Left target
                p = l - 1
                if p >= 0 and output[r][p] != 2:
                    targets_h.append(p)
                # Right target
                p = rr + 1
                if p < w and output[r][p] != 2:
                    targets_h.append(p)
            else:
                i += 1
        dists_h = [(abs(c - p), p) for p in targets_h]
        min_dh = min([d[0] for d in dists_h] + [float('inf')])
        candidates_h = [p for d, p in dists_h if d == min_dh]

        # Vertical targets in col c
        targets_v = []
        i = 0
        while i < h:
            if output[i][c] == 2:
                t = i
                while i < h and output[i][c] == 2:
                    i += 1
                b = i - 1
                # Above target
                q = t - 1
                if q >= 0 and output[q][c] != 2:
                    targets_v.append(q)
                # Below target
                q = b + 1
                if q < h and output[q][c] != 2:
                    targets_v.append(q)
            else:
                i += 1
        dists_v = [(abs(r - q), q) for q in targets_v]
        min_dv = min([d[0] for d in dists_v] + [float('inf')])
        candidates_v = [q for d, q in dists_v if d == min_dv]

        target = None
        do_move = False
        if min_dh < float('inf') and min_dh <= min_dv:
            if min_dh > 0:
                # Choose the target with min dist, then min |c-p|, then min p
                best = min(candidates_h, key=lambda p: (abs(c - p), p))
                target = (r, best)
                do_move = True
        elif min_dv < float('inf'):
            if min_dv > 0:
                # Similarly for vertical
                best = min(candidates_v, key=lambda q: (abs(r - q), q))
                target = (best, c)
                do_move = True

        if do_move and target:
            tr, tc = target
            if target not in to_move:
                to_move[target] = []
            to_move[target].append((r, c))

    # Apply moves
    for orig in ones:
        moved = False
        for t, origs in to_move.items():
            if orig in origs:
                moved = True
                break
        if moved:
            output[orig[0]][orig[1]] = 0

    for targ in to_move:
        output[targ[0]][targ[1]] = 1

    return output