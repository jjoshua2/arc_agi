from collections import Counter
from typing import List

def transform(grid_lst: List[List[int]]) -> List[List[int]]:
    grid = grid_lst
    h = len(grid)
    w = len(grid[0]) if h > 0 else 0
    if h == 0 or w == 0:
        return []

    # Vertical boundaries
    vert_scores = [0] * (w - 1)
    for j in range(1, w):
        score = sum(1 for i in range(h) if grid[i][j - 1] != grid[i][j])
        vert_scores[j - 1] = score

    threshold_v = h * 0.5
    vert_bound = [0]
    for j in range(1, w):
        if vert_scores[j - 1] > threshold_v:
            vert_bound.append(j)
    vert_bound.append(w)

    # Horizontal boundaries
    horiz_scores = [0] * (h - 1)
    for i in range(1, h):
        score = sum(1 for j in range(w) if grid[i - 1][j] != grid[i][j])
        horiz_scores[i - 1] = score

    threshold_h = w * 0.5
    horiz_bound = [0]
    for i in range(1, h):
        if horiz_scores[i - 1] > threshold_h:
            horiz_bound.append(i)
    horiz_bound.append(h)

    # Output dimensions
    out_h = len(horiz_bound) - 1
    out_w = len(vert_bound) - 1

    output = [[0] * out_w for _ in range(out_h)]

    for hi in range(out_h):
        r_start = horiz_bound[hi]
        r_end = horiz_bound[hi + 1] - 1
        for vj in range(out_w):
            c_start = vert_bound[vj]
            c_end = vert_bound[vj + 1] - 1
            flat = [grid[rr][cc] for rr in range(r_start, r_end + 1) for cc in range(c_start, c_end + 1)]
            if flat:
                cnt = Counter(flat)
                output[hi][vj] = cnt.most_common(1)[0][0]
            else:
                output[hi][vj] = 1  # Default to 1 if empty (unlikely)

    return output