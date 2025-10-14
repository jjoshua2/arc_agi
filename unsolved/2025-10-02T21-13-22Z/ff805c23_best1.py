def transform(grid: list[list[int]]) -> list[list[int]]:
    if not grid or not grid[0]:
        return []
    n = 24  # Fixed size
    mismatches = []
    for r in range(12):
        b = 23 - r
        for c in range(n):
            if grid[r][c] != grid[b][c]:
                mismatches.append((b, c))
    if not mismatches:
        return [row[:] for row in grid]
    paint_colors = {grid[b][c] for b, c in mismatches}
    if len(paint_colors) != 1:
        return [row[:] for row in grid]
    bs = [b for b, c in mismatches]
    cs = [c for b, c in mismatches]
    min_b = min(bs)
    max_b = max(bs)
    min_c = min(cs)
    max_c = max(cs)
    h = max_b - min_b + 1
    w = max_c - min_c + 1
    num_mis = len(mismatches)
    if num_mis != h * w:
        return [row[:] for row in grid]
    # Build output
    output = []
    for i in range(h):
        b = min_b + i
        top_r = 23 - b
        row = [grid[top_r][min_c + j] for j in range(w)]
        output.append(row)
    return output