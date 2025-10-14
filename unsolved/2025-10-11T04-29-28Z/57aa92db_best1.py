import numpy as np

def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    grid = np.array(grid_lst, dtype=int)
    rows, cols = grid.shape
    out = grid.copy()

    # Find connected components of non-zero cells (4-connected)
    visited = np.zeros_like(grid, dtype=bool)
    components = []  # list of dicts: {id, color, min_r, max_r, min_c, max_c, s}
    comp_id = 0

    for r in range(rows):
        for c in range(cols):
            if grid[r, c] != 0 and not visited[r, c]:
                color = int(grid[r, c])
                # BFS/DFS to collect the component
                stack = [(r, c)]
                visited[r, c] = True
                cells = []
                while stack:
                    rr, cc = stack.pop()
                    cells.append((rr, cc))
                    for dr, dc in ((-1,0),(1,0),(0,-1),(0,1)):
                        nr, nc = rr + dr, cc + dc
                        if 0 <= nr < rows and 0 <= nc < cols and (not visited[nr, nc]) and grid[nr, nc] == color:
                            visited[nr, nc] = True
                            stack.append((nr, nc))
                rs = [p[0] for p in cells]
                cs = [p[1] for p in cells]
                min_r, max_r = min(rs), max(rs)
                min_c, max_c = min(cs), max(cs)
                height = max_r - min_r + 1
                width = max_c - min_c + 1
                # accept only solid filled squares (all cells inside bounding box are the component)
                if height == width and len(cells) == height * width:
                    s = height
                    components.append({
                        "id": comp_id,
                        "color": color,
                        "min_r": min_r,
                        "max_r": max_r,
                        "min_c": min_c,
                        "max_c": max_c,
                        "s": s
                    })
                    comp_id += 1
                # else ignore non-square components

    # Find adjacent square pairs (horizontal or vertical), and expand the dominant color
    used_pairs = set()
    n = len(components)
    for i in range(n):
        for j in range(i+1, n):
            a = components[i]
            b = components[j]
            # require same size
            if a["s"] != b["s"]:
                continue
            s = a["s"]

            # check horizontal adjacency: same rows (min_r equal) and contiguous columns
            if a["min_r"] == b["min_r"] and a["max_r"] == b["max_r"]:
                # contiguous horizontally?
                if a["min_c"] + s == b["min_c"] or b["min_c"] + s == a["min_c"]:
                    # determine left and right
                    if a["min_c"] < b["min_c"]:
                        left, right = a, b
                    else:
                        left, right = b, a

                    pair_key = ("h", left["min_r"], left["min_c"], left["id"], right["id"])
                    if pair_key in used_pairs:
                        continue
                    used_pairs.add(pair_key)

                    # choose dominant color (higher numeric). Tie-break deterministically: left.
                    if left["color"] > right["color"]:
                        dominant = left
                        dominant_side = "left"
                    elif right["color"] > left["color"]:
                        dominant = right
                        dominant_side = "right"
                    else:
                        dominant = left
                        dominant_side = "left"

                    # compute expansion top-left (row_exp, col_exp) and fill 3*s x 3*s region
                    row_exp = left["min_r"] - s
                    if dominant_side == "left":
                        col_exp = left["min_c"] - s
                    else:
                        col_exp = left["min_c"]

                    # Fill only zeros, and only inside bounds
                    for rr in range(row_exp, row_exp + 3*s):
                        if rr < 0 or rr >= rows:
                            continue
                        for cc in range(col_exp, col_exp + 3*s):
                            if cc < 0 or cc >= cols:
                                continue
                            if out[rr, cc] == 0:
                                out[rr, cc] = dominant["color"]

            # check vertical adjacency: same columns (min_c equal) and contiguous rows
            if a["min_c"] == b["min_c"] and a["max_c"] == b["max_c"]:
                # contiguous vertically?
                if a["min_r"] + s == b["min_r"] or b["min_r"] + s == a["min_r"]:
                    # determine top and bottom
                    if a["min_r"] < b["min_r"]:
                        top, bottom = a, b
                    else:
                        top, bottom = b, a

                    pair_key = ("v", top["min_r"], top["min_c"], top["id"], bottom["id"])
                    if pair_key in used_pairs:
                        continue
                    used_pairs.add(pair_key)

                    # dominant: higher numeric, tie-break top
                    if top["color"] > bottom["color"]:
                        dominant = top
                        dominant_side = "top"
                    elif bottom["color"] > top["color"]:
                        dominant = bottom
                        dominant_side = "bottom"
                    else:
                        dominant = top
                        dominant_side = "top"

                    # compute expansion top-left
                    col_exp = top["min_c"] - s
                    if dominant_side == "top":
                        row_exp = top["min_r"]
                    else:
                        row_exp = top["min_r"] - s

                    # Fill only zeros, and only inside bounds
                    for rr in range(row_exp, row_exp + 3*s):
                        if rr < 0 or rr >= rows:
                            continue
                        for cc in range(col_exp, col_exp + 3*s):
                            if cc < 0 or cc >= cols:
                                continue
                            if out[rr, cc] == 0:
                                out[rr, cc] = dominant["color"]

    return out.tolist()