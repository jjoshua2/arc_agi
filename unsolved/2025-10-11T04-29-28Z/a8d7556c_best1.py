from typing import List

def transform(grid: List[List[int]]) -> List[List[int]]:
    if not grid or not grid[0]:
        return []
    rows = len(grid)
    cols = len(grid[0])
    # Work from the original grid for detection
    orig = grid
    # Make a copy for output
    out = [row[:] for row in orig]
    # Scan all possible top-left corners of 2x2 subgrids
    for r in range(rows - 1):
        for c in range(cols - 1):
            if orig[r][c] == 0 and orig[r][c+1] == 0 and orig[r+1][c] == 0 and orig[r+1][c+1] == 0:
                out[r][c]     = 2
                out[r][c+1]   = 2
                out[r+1][c]   = 2
                out[r+1][c+1] = 2
    return out