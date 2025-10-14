def transform(grid: list[list[int]]) -> list[list[int]]:
    if not grid or not grid[0]:
        return [row[:] for row in grid]
    h, w = len(grid), len(grid[0])
    # Work on a copy for the result
    out = [row[:] for row in grid]

    to_fill = set()  # coordinates to recolor to 8

    # Helper to check if all cells in a row segment are 5
    def all_five_row(r: int, c1: int, c2: int) -> bool:
        # assumes c1 < c2
        for c in range(c1 + 1, c2):
            if grid[r][c] != 5:
                return False
        return True

    # Helper to check if all cells in a column segment are 5
    def all_five_col(c: int, r1: int, r2: int) -> bool:
        # assumes r1 < r2
        for r in range(r1 + 1, r2):
            if grid[r][c] != 5:
                return False
        return True

    # Process rows: fill gaps of 5s between consecutive red (2) cells
    for r in range(h):
        red_cols = [c for c in range(w) if grid[r][c] == 2]
        for i in range(len(red_cols) - 1):
            c1, c2 = red_cols[i], red_cols[i + 1]
            if c2 - c1 > 1 and all_five_row(r, c1, c2):
                for c in range(c1 + 1, c2):
                    to_fill.add((r, c))

    # Process columns: fill gaps of 5s between consecutive red (2) cells
    for c in range(w):
        red_rows = [r for r in range(h) if grid[r][c] == 2]
        for i in range(len(red_rows) - 1):
            r1, r2 = red_rows[i], red_rows[i + 1]
            if r2 - r1 > 1 and all_five_col(c, r1, r2):
                for r in range(r1 + 1, r2):
                    to_fill.add((r, c))

    # Apply fills
    for r, c in to_fill:
        # Only 5s turn to 8; other cells stay as they are
        if out[r][c] == 5:
            out[r][c] = 8

    return out