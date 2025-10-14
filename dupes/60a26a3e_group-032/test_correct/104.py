from collections import defaultdict
from typing import List, Tuple

def transform(grid: List[List[int]]) -> List[List[int]]:
    h = len(grid)
    if h == 0:
        return []
    w = len(grid[0])
    output = [row[:] for row in grid]
    
    centers: List[Tuple[int, int]] = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if (grid[r][c] == 0 and
                grid[r][c - 1] == 2 and
                grid[r][c + 1] == 2 and
                grid[r - 1][c] == 2 and
                grid[r + 1][c] == 2):
                centers.append((r, c))
    
    row_to_cs = defaultdict(list)
    col_to_rs = defaultdict(list)
    for r, c in centers:
        row_to_cs[r].append(c)
        col_to_rs[c].append(r)
    
    # Horizontal connections
    for r, cs in row_to_cs.items():
        if len(cs) > 1:
            cs.sort()
            for i in range(len(cs) - 1):
                c1 = cs[i]
                c2 = cs[i + 1]
                start_col = c1 + 2
                end_col = c2 - 2
                if start_col <= end_col:
                    for col in range(start_col, end_col + 1):
                        output[r][col] = 1
    
    # Vertical connections
    for c, rs in col_to_rs.items():
        if len(rs) > 1:
            rs.sort()
            for i in range(len(rs) - 1):
                r1 = rs[i]
                r2 = rs[i + 1]
                start_row = r1 + 2
                end_row = r2 - 2
                if start_row <= end_row:
                    for row in range(start_row, end_row + 1):
                        output[row][c] = 1
    
    return output