from collections import defaultdict
from typing import List

def transform(grid_lst: List[List[int]]) -> List[List[int]]:
    if not grid_lst:
        return []
    grid = [row[:] for row in grid_lst]
    rows = len(grid)
    if rows == 0:
        return grid
    cols_num = len(grid[0])
    centers = []
    for i in range(rows):
        for j in range(cols_num):
            if grid[i][j] != 0:
                continue
            has_up = i > 0 and grid[i-1][j] == 2
            has_down = i + 1 < rows and grid[i + 1][j] == 2
            has_left = j > 0 and grid[i][j - 1] == 2
            has_right = j + 1 < cols_num and grid[i][j + 1] == 2
            if has_up and has_down and has_left and has_right:
                centers.append((i, j))
    # Horizontal connections
    horiz = defaultdict(list)
    for r, c in centers:
        horiz[r].append(c)
    for r, clst in horiz.items():
        if len(clst) > 1:
            clst.sort()
            for k in range(len(clst) - 1):
                j1 = clst[k]
                j2 = clst[k + 1]
                if j2 > j1 + 2:
                    sc = j1 + 2
                    ec = j2 - 2
                    for cc in range(sc, ec + 1):
                        if 0 <= cc < cols_num:
                            grid[r][cc] = 1
    # Vertical connections
    vert = defaultdict(list)
    for r, c in centers:
        vert[c].append(r)
    for c, rlst in vert.items():
        if len(rlst) > 1:
            rlst.sort()
            for k in range(len(rlst) - 1):
                i1 = rlst[k]
                i2 = rlst[k + 1]
                if i2 > i1 + 2:
                    sr = i1 + 2
                    er = i2 - 2
                    for rr in range(sr, er + 1):
                        if 0 <= rr < rows:
                            grid[rr][c] = 1
    return grid