from typing import List, Tuple
import math

def transform(grid: List[List[int]]) -> List[List[int]]:
    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0
    out = [[0 for _ in range(cols)] for _ in range(rows)]
    
    # 1) Find the unique nonzero color (the dots)
    pts: List[Tuple[int,int,int]] = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 0:
                pts.append((r, c, grid[r][c]))
    if not pts:
        return out
    # assume all nonzero are the same color (as in the task)
    color = pts[0][2]
    seed_positions = [(r,c) for r,c,_ in pts if _ == color]
    
    # Keep only up to 3 seeds if more happen to exist
    if len(seed_positions) > 3:
        seed_positions = seed_positions[:3]
    
    # 2) Compute spacing s using gcd on differences in rows and columns
    row_diffs = []
    col_diffs = []
    for i in range(len(seed_positions)):
        for j in range(i+1, len(seed_positions)):
            dr = abs(seed_positions[i][0] - seed_positions[j][0])
            dc = abs(seed_positions[i][1] - seed_positions[j][1])
            if dr != 0:
                row_diffs.append(dr)
            if dc != 0:
                col_diffs.append(dc)
    # Default to 1 if something goes wrong
    s_row = 0
    for d in row_diffs:
        s_row = d if s_row == 0 else math.gcd(s_row, d)
    s_col = 0
    for d in col_diffs:
        s_col = d if s_col == 0 else math.gcd(s_col, d)
    s_candidates = [v for v in [s_row, s_col] if v != 0]
    s = min(s_candidates) if s_candidates else 1
    if s <= 0:
        s = 1
    
    # 3) Modular offsets from any seed
    sr, sc = seed_positions[0]
    r_rem = sr % s
    c_rem = sc % s
    
    # 4) Determine the outermost ring aligned by those remainders
    def first_with_remainder(limit: int, rem: int, step: int) -> int:
        # smallest x in [0..limit] such that x % step == rem
        r0 = rem
        if r0 < 0:
            r0 += step
        while r0 < 0:
            r0 += step
        while r0 > limit:
            r0 -= step
        # If above loop overshot (rare), clamp using standard formula:
        if r0 < 0: 
            r0 = rem % step
        return r0
    
    def last_with_remainder(limit: int, rem: int, step: int) -> int:
        # largest x in [0..limit] such that x % step == rem
        if limit < 0:
            return -1
        # compute base remainder <= limit
        # Let k = floor((limit - rem)/step)
        k = (limit - rem) // step
        return rem + k * step
    
    top = first_with_remainder(rows - 1, r_rem, s)
    left = first_with_remainder(cols - 1, c_rem, s)
    bottom = last_with_remainder(rows - 1, r_rem, s)
    right = last_with_remainder(cols - 1, c_rem, s)
    
    # 5) Draw the spiral rings
    # We ensure indices remain valid and draw with gaps of width s-1 on top/right
    gap = s - 1
    
    while top <= bottom and left <= right:
        # top segment: from left to (right - gap - 1), inclusive
        end_top = right - gap - 1
        if end_top >= left and 0 <= top < rows:
            for c in range(max(0, left), min(cols - 1, end_top) + 1):
                out[top][c] = color
        
        # right segment: from top to (bottom - gap - 1), inclusive
        end_right = bottom - gap - 1
        if end_right >= top and 0 <= right < cols:
            for r in range(max(0, top), min(rows - 1, end_right) + 1):
                out[r][right] = color
        
        # bottom segment: full from left to right, inclusive (if bottom is valid)
        if 0 <= bottom < rows:
            for c in range(max(0, left), min(cols - 1, right) + 1):
                out[bottom][c] = color
        
        # left segment: from top to (bottom - gap - 1), inclusive
        if end_right >= top and 0 <= left < cols:
            for r in range(max(0, top), min(rows - 1, end_right) + 1):
                out[r][left] = color
        
        # shrink the ring by s
        top += s
        left += s
        bottom -= s
        right -= s
    
    # 6) If there is a degenerate central cell to mark
    if 0 <= top <= rows - 1 and 0 <= left <= cols - 1 and top == bottom and left == right:
        out[top][left] = color
    
    return out