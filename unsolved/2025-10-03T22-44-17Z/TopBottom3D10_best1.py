def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    if not grid_lst or not grid_lst[0]:
        return []
    h = len(grid_lst)
    w = len(grid_lst[0])
    grid = [row[:] for row in grid_lst]

    # Find all positions with color 8 (the central frame)
    pos8 = [(r, c) for r in range(h) for c in range(w) if grid[r][c] == 8]
    if not pos8:
        return grid

    min_r = min(r for r, _ in pos8)
    max_r = max(r for r, _ in pos8)
    min_c = min(c for _, c in pos8)
    max_c = max(c for _, c in pos8)

    # If the interior region is empty (frame too thin), nothing to map into.
    interior_r_min = min_r + 1
    interior_r_max = max_r - 1
    interior_c_min = min_c + 1
    interior_c_max = max_c - 1
    interior_exists = (interior_r_min <= interior_r_max) and (interior_c_min <= interior_c_max)

    if not interior_exists:
        return grid

    # Helper to set interior cell if in bounds
    def set_interior(tr, tc, color):
        if 0 <= tr < h and 0 <= tc < w:
            grid[tr][tc] = color

    # Iterate all cells outside the frame
    for r in range(h):
        for c in range(w):
            color = grid_lst[r][c]
            if color == 0 or color == 8:
                continue

            target = None
            # Side mappings
            if interior_exists:
                if min_r <= r <= max_r and c < min_c:
                    target = (r, min_c + 1)
                elif min_r <= r <= max_r and c > max_c:
                    target = (r, max_c - 1)
                elif min_c <= c <= max_c and r < min_r:
                    target = (min_r + 1, c)
                elif min_c <= c <= max_c and r > max_r:
                    target = (max_r - 1, c)
                else:
                    # Diagonal/outside around corners
                    if r < min_r and c < min_c:
                        target = (min_r + 1, min_c + 1)
                    elif r < min_r and c > max_c:
                        target = (min_r + 1, max_c - 1)
                    elif r > max_r and c < min_c:
                        target = (max_r - 1, min_c + 1)
                    elif r > max_r and c > max_c:
                        target = (max_r - 1, max_c - 1)

            if target is not None:
                tr, tc = target
                set_interior(tr, tc, color)

    return grid