def transform(grid: list[list[int]]) -> list[list[int]]:
    if not grid or not grid[0]:
        return []
    rows = len(grid)
    cols = len(grid[0])
    # Find bar rows
    bar_rows = [r for r in range(rows) if any(cell == 5 for cell in grid[r])]
    if not bar_rows:
        return [row[:] for row in grid]
    min_r = min(bar_rows)
    max_r = max(bar_rows)
    # Take pattern from first bar row
    bar_pattern = grid[min_r]
    # Find min_col, max_col
    five_cols = [c for c in range(cols) if bar_pattern[c] == 5]
    if not five_cols:
        return [row[:] for row in grid]
    min_col = min(five_cols)
    max_col = max(five_cols)
    internal_gaps = [c for c in range(cols) if min_col <= c <= max_col and bar_pattern[c] == 0]
    # Output
    output = [row[:] for row in grid]
    for r in range(rows):
        if any(cell == 5 for cell in grid[r]):
            # bar row
            for c in range(cols):
                if grid[r][c] == 5:
                    output[r][c] = 5
                elif min_col <= c <= max_col and grid[r][c] == 0:
                    output[r][c] = 2
                else:
                    output[r][c] = grid[r][c]  # probably 0
        else:
            # empty row
            if r < min_r or r > max_r:
                # top or bottom
                for c in internal_gaps:
                    output[r][c] = 1
            else:
                # inter
                for c in range(cols):
                    if c < min_col or c > max_col:
                        output[r][c] = 1
                    else:
                        output[r][c] = 2
    return output