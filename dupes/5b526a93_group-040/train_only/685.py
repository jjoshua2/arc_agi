from collections import defaultdict
from typing import List

def transform(grid_lst: List[List[int]]) -> List[List[int]]:
    if not grid_lst or not grid_lst[0]:
        return [row[:] for row in grid_lst]
    h = len(grid_lst)
    w = len(grid_lst[0])
    if h < 3 or w < 3:
        return [row[:] for row in grid_lst]

    def is_ring(grid, r: int, c: int, col: int) -> bool:
        # Top row
        if not (0 <= c < w - 2 and 0 <= c + 2 < w):
            return False
        if grid[r][c] != col or grid[r][c + 1] != col or grid[r][c + 2] != col:
            return False
        # Middle row
        if grid[r + 1][c] != col or grid[r + 1][c + 1] != 0 or grid[r + 1][c + 2] != col:
            return False
        # Bottom row
        if grid[r + 2][c] != col or grid[r + 2][c + 1] != col or grid[r + 2][c + 2] != col:
            return False
        return True

    ring_groups = defaultdict(list)
    for r in range(h - 2):
        for c in range(w - 2):
            if is_ring(grid_lst, r, c, 1):
                ring_groups[r].append(c)

    if not ring_groups:
        return [row[:] for row in grid_lst]

    # Find r_ref with max rings
    r_ref = max(ring_groups, key=lambda k: len(ring_groups[k]))
    targets = set(ring_groups[r_ref])

    output = [row[:] for row in grid_lst]

    for r in ring_groups:
        original_cs = set(ring_groups[r])
        for t in targets:
            if t not in original_cs:
                # Place 8-ring
                for j in range(3):
                    output[r][t + j] = 8
                    output[r + 2][t + j] = 8
                output[r + 1][t] = 8
                output[r + 1][t + 2] = 8

    return output