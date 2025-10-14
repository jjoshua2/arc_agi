from collections import deque
from typing import List

def transform(grid: List[List[int]]) -> List[List[int]]:
    if not grid or not grid[0]:
        return []
    rows = len(grid)
    cols = len(grid[0])
    output = [row[:] for row in grid]
    
    # First, cap fills (above horizontals) from bottom to top
    for r in range(rows - 1, -1, -1):
        for c in range(cols):
            if output[r][c] != 0:
                continue
            left_ok = (c == 0 or output[r][c - 1] in (2, 8))
            below_ok = (r + 1 < rows and output[r + 1][c] in (2, 8))
            if left_ok and below_ok:
                # Compute gap
                next_b = c
                while next_b < cols and output[r][next_b] == 0:
                    next_b += 1
                gap = next_b - c
                if next_b < cols and gap > 3:
                    continue  # large closed gap, skip
                # Fill the gap with 8
                for k in range(c, next_b):
                    output[r][k] = 8
    
    # Then, bay fills (below horizontals) from top to bottom
    for r in range(rows):
        for c in range(1, cols):
            if output[r][c] != 0:
                continue
            left_ok = (output[r][c - 1] == 2)
            above_ok = (r == 0 or output[r - 1][c] in (2, 8))
            if left_ok and above_ok:
                # Compute gap
                next_b = c
                while next_b < cols and output[r][next_b] == 0:
                    next_b += 1
                gap = next_b - c
                if gap <= 3:
                    # Small gap: fill single row with 8, no flood
                    for k in range(c, next_b):
                        output[r][k] = 8
                else:
                    # Large gap: flood fill down and right with 1
                    q = deque()
                    if output[r][c] == 0:
                        output[r][c] = 1
                        q.append((r, c))
                    while q:
                        rr, cc = q.popleft()
                        # right
                        if cc + 1 < cols and output[rr][cc + 1] == 0:
                            output[rr][cc + 1] = 1
                            q.append((rr, cc + 1))
                        # down
                        if rr + 1 < rows and output[rr + 1][cc] == 0:
                            output[rr + 1][cc] = 1
                            q.append((rr + 1, cc))
    
    return output