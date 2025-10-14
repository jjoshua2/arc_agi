def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    if not grid_lst or not grid_lst[0]:
        return []
    rows = len(grid_lst)
    cols = len(grid_lst[0])
    positions = []
    for r in range(rows):
        for c in range(cols):
            if grid_lst[r][c] != 0:
                positions.append((r, c, grid_lst[r][c]))
    if len(positions) != 2:
        return [row[:] for row in grid_lst]  # If not exactly two, return unchanged (though examples have two)
    (r1, c1, color1), (r2, c2, color2) = positions
    output = [[0 for _ in range(cols)] for _ in range(rows)]
    if rows > cols:  # Vertical pattern
        if r1 > r2:
            r1, c1, color1, r2, c2, color2 = r2, c2, color2, r1, c1, color1
        step = r2 - r1
        current_row = r1
        current_color = color1
        next_color = color2
        while current_row < rows:
            for c in range(cols):
                output[current_row][c] = current_color
            current_row += step
            current_color, next_color = next_color, current_color
    else:  # Horizontal pattern
        if c1 > c2:
            r1, c1, color1, r2, c2, color2 = r2, c2, color2, r1, c1, color1
        step = c2 - c1
        current_col = c1
        current_color = color1
        next_color = color2
        while current_col < cols:
            for r in range(rows):
                output[r][current_col] = current_color
            current_col += step
            current_color, next_color = next_color, current_color
    return output