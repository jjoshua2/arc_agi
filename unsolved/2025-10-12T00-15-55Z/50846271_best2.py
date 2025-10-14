from typing import List, Set, Tuple

def transform(grid: List[List[int]]) -> List[List[int]]:
    if not grid or not grid[0]:
        return []
    rows = len(grid)
    cols = len(grid[0])
    change: Set[Tuple[int, int]] = set()

    # Horizontal gap filling
    for r in range(rows):
        pos: List[int] = [c for c in range(cols) if grid[r][c] == 2]
        if len(pos) >= 2:
            min_c = min(pos)
            max_c = max(pos)
            for c in range(min_c + 1, max_c):
                if grid[r][c] == 5:
                    change.add((r, c))

    # Vertical gap filling
    for c in range(cols):
        pos: List[int] = [r for r in range(rows) if grid[r][c] == 2]
        if len(pos) >= 2:
            min_r = min(pos)
            max_r = max(pos)
            for r in range(min_r + 1, max_r):
                if grid[r][c] == 5:
                    change.add((r, c))

    # Local extension rule
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 5:
                continue
            n2 = 0
            n0 = 0
            n5 = 0
            for dr, dc in dirs:
                nr = r + dr
                nc = c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    val = grid[nr][nc]
                    if val == 2:
                        n2 += 1
                    elif val == 0:
                        n0 += 1
                    elif val == 5:
                        n5 += 1
            if n2 == 1 and n0 == 2 and n5 == 1:
                change.add((r, c))

    # Apply changes
    out = [row[:] for row in grid]
    for rr, cc in change:
        out[rr][cc] = 8
    return out