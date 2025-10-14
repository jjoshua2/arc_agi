def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    if not grid_lst or not grid_lst[0]:
        return []
    h_grid = len(grid_lst)
    w_grid = len(grid_lst[0])
    candidates_high = []
    candidates_all = []
    for C in range(1, 10):
        positions = [(r, c) for r in range(h_grid) for c in range(w_grid) if grid_lst[r][c] == C]
        if not positions:
            continue
        min_r = min(r for r, c in positions)
        max_r = max(r for r, c in positions)
        min_c = min(c for r, c in positions)
        max_c = max(c for r, c in positions)
        area = (max_r - min_r + 1) * (max_c - min_c + 1)
        info = (area, C, min_r, max_r, min_c, max_c)
        candidates_all.append(info)
        if C >= 6:
            candidates_high.append(info)
    if candidates_high:
        candidates_high.sort(key=lambda x: (-x[0], -x[1]))
        selected = candidates_high[0]
    else:
        if not candidates_all:
            return []
        candidates_all.sort(key=lambda x: (-x[0], -x[1]))
        selected = candidates_all[0]
    _, C, min_r, max_r, min_c, max_c = selected
    h_out = max_r - min_r + 1
    w_out = max_c - min_c + 1
    output = [[0] * w_out for _ in range(h_out)]
    for r in range(h_grid):
        for c in range(w_grid):
            if grid_lst[r][c] == C:
                rel_r = r - min_r
                rel_c = c - min_c
                output[rel_r][rel_c] = C
    return output