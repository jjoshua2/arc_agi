from typing import List, Tuple
from math import gcd
from collections import Counter
import copy

def _gcd_list(vals):
    g = 0
    for v in vals:
        g = gcd(g, abs(v))
    return g

def _spiral_order(m: int, n: int) -> List[Tuple[int,int]]:
    # clockwise spiral traversal of an m x n grid starting at (0,0) going right
    res = []
    top, left = 0, 0
    bottom, right = m - 1, n - 1
    while top <= bottom and left <= right:
        # top row
        for j in range(left, right + 1):
            res.append((top, j))
        top += 1
        # right column
        for i in range(top, bottom + 1):
            res.append((i, right))
        right -= 1
        if top <= bottom:
            # bottom row (right->left)
            for j in range(right, left - 1, -1):
                res.append((bottom, j))
            bottom -= 1
        if left <= right:
            # left column (bottom->top)
            for i in range(bottom, top - 1, -1):
                res.append((i, left))
            left += 1
    return res

def transform(grid: List[List[int]]) -> List[List[int]]:
    if not grid or not grid[0]:
        return grid
    rows = len(grid)
    cols = len(grid[0])

    # find the dominant non-zero color (seed color)
    color_counts = Counter()
    positions = {}
    for r in range(rows):
        for c in range(cols):
            v = grid[r][c]
            if v != 0:
                color_counts[v] += 1
                positions.setdefault(v, []).append((r, c))
    if not color_counts:
        return [row[:] for row in grid]

    # choose the non-zero color with the largest count (robustness)
    target_color = max(color_counts.items(), key=lambda x: (x[1], x[0]))[0]
    seeds = positions[target_color]

    # If seeds are fewer than 2, nothing clear to do; just return a grid of that color at seed positions.
    if len(seeds) < 2:
        out = [[0]*cols for _ in range(rows)]
        for (r,c) in seeds:
            out[r][c] = target_color
        return out

    # compute spacing S from differences of seed positions (use gcd of all row diffs and col diffs)
    seeds_sorted_by_row = sorted(seeds)
    row_diffs = []
    col_diffs = []
    for i in range(len(seeds_sorted_by_row)-1):
        r1,c1 = seeds_sorted_by_row[i]
        r2,c2 = seeds_sorted_by_row[i+1]
        row_diffs.append(r2 - r1)
        col_diffs.append(c2 - c1)
    # also include differences across all pairs just in case
    # but gcd of adjacent sorted should be fine; take gcd of both row and col diffs
    g_row = _gcd_list(row_diffs) if row_diffs else 0
    g_col = _gcd_list(col_diffs) if col_diffs else 0
    S = gcd(g_row, g_col)
    if S == 0:
        # fallback: try absolute differences among all pairs
        all_row_diffs = []
        all_col_diffs = []
        for i in range(len(seeds)):
            for j in range(i+1, len(seeds)):
                all_row_diffs.append(abs(seeds[i][0] - seeds[j][0]))
                all_col_diffs.append(abs(seeds[i][1] - seeds[j][1]))
        S = gcd(_gcd_list(all_row_diffs), _gcd_list(all_col_diffs)) if all_row_diffs or all_col_diffs else 1
    if S <= 0:
        S = 1

    # determine modal residues for row and column modulo S (robust to ordering)
    row_mods = [r % S for (r, _) in seeds]
    col_mods = [c % S for (_, c) in seeds]
    rmod = Counter(row_mods).most_common(1)[0][0]
    cmod = Counter(col_mods).most_common(1)[0][0]

    # compute lists of fine-grid coordinates that lie on the S-grid
    rcoords = [r for r in range(rows) if r % S == rmod]
    ccoords = [c for c in range(cols) if c % S == cmod]

    if not rcoords or not ccoords:
        # fallback: if something went wrong, just return original grid
        return [row[:] for row in grid]

    m = len(rcoords)
    n = len(ccoords)

    # get coarse spiral sequence
    coarse_seq = _spiral_order(m, n)
    # map coarse to fine coords
    fine_seq = [(rcoords[i], ccoords[j]) for (i,j) in coarse_seq]

    # prepare output (start with all zeros)
    out = [[0]*cols for _ in range(rows)]

    def draw_line(p1, p2):
        r1,c1 = p1
        r2,c2 = p2
        if r1 == r2:
            if c1 <= c2:
                for cc in range(c1, c2+1):
                    out[r1][cc] = target_color
            else:
                for cc in range(c2, c1+1):
                    out[r1][cc] = target_color
        elif c1 == c2:
            if r1 <= r2:
                for rr in range(r1, r2+1):
                    out[rr][c1] = target_color
            else:
                for rr in range(r2, r1+1):
                    out[rr][c1] = target_color
        else:
            # should not occur for adjacent coarse points, but handle with L-shape:
            # go horizontally then vertically
            step = 1 if c2 >= c1 else -1
            for cc in range(c1, c2 + step, step):
                out[r1][cc] = target_color
            step2 = 1 if r2 >= r1 else -1
            for rr in range(r1, r2 + step2, step2):
                out[rr][c2] = target_color

    # draw segments between successive fine points
    if len(fine_seq) == 1:
        # single coarse point - color that fine point and extend to nearest boundaries
        r0,c0 = fine_seq[0]
        out[r0][c0] = target_color
    else:
        for k in range(len(fine_seq) - 1):
            draw_line(fine_seq[k], fine_seq[k+1])

        # extend the first point perpendicularly to the nearest boundary
        f0 = fine_seq[0]
        f1 = fine_seq[1]
        dr = f1[0] - f0[0]
        dc = f1[1] - f0[1]
        if dr == 0:
            # movement is horizontal; extend vertically from f0
            dist_top = f0[0] - 0
            dist_bottom = (rows - 1) - f0[0]
            if dist_top <= dist_bottom:
                for rr in range(0, f0[0] + 1):
                    out[rr][f0[1]] = target_color
            else:
                for rr in range(f0[0], rows):
                    out[rr][f0[1]] = target_color
        else:
            # movement vertical; extend horizontally from f0
            dist_left = f0[1] - 0
            dist_right = (cols - 1) - f0[1]
            if dist_left <= dist_right:
                for cc in range(0, f0[1] + 1):
                    out[f0[0]][cc] = target_color
            else:
                for cc in range(f0[1], cols):
                    out[f0[0]][cc] = target_color

        # extend the last point perpendicularly to nearest boundary
        fl = fine_seq[-1]
        flm1 = fine_seq[-2]
        dr = fl[0] - flm1[0]
        dc = fl[1] - flm1[1]
        if dr == 0:
            # horizontal movement -> extend vertically from fl
            dist_top = fl[0] - 0
            dist_bottom = (rows - 1) - fl[0]
            if dist_top <= dist_bottom:
                for rr in range(0, fl[0] + 1):
                    out[rr][fl[1]] = target_color
            else:
                for rr in range(fl[0], rows):
                    out[rr][fl[1]] = target_color
        else:
            # vertical movement -> extend horizontally from fl
            dist_left = fl[1] - 0
            dist_right = (cols - 1) - fl[1]
            if dist_left <= dist_right:
                for cc in range(0, fl[1] + 1):
                    out[fl[0]][cc] = target_color
            else:
                for cc in range(fl[1], cols):
                    out[fl[0]][cc] = target_color

    return out