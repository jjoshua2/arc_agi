def transform(grid: list[list[int]]) -> list[list[int]]:
    if not grid or not grid[0]:
        return []
    rows = len(grid)
    cols = len(grid[0])
    output = [row[:] for row in grid]
    # Collect palette from left vertical strips
    palette = []
    for c in range(cols):
        non_zero_colors = [grid[r][c] for r in range(rows) if grid[r][c] != 0]
        if non_zero_colors:
            color = non_zero_colors[0]
            if all(x == color for x in non_zero_colors) and color != 8 and color > 0:
                palette.append(color)
                # Clear the palette column in output
                for r in range(rows):
                    output[r][c] = 0
    # Fill purple groups
    palette_idx = 0
    c = 0
    while c < cols:
        # Find the run in this column using input grid
        run_start = -1
        run_end = -1
        r = 0
        while r < rows:
            if grid[r][c] == 8:
                run_start = r
                run_end = r
                r += 1
                while r < rows and grid[r][c] == 8:
                    run_end = r
                    r += 1
                # Assume no more runs after this; break or continue as needed
                break
            r += 1
        if run_start == -1:
            c += 1
            continue
        # Extend to group of consecutive columns with exact same run
        group_end = c
        for cc in range(c + 1, cols):
            match = True
            for rr in range(rows):
                expected = 8 if run_start <= rr <= run_end else 0
                if grid[rr][cc] != expected:
                    match = False
                    break
            if match:
                group_end = cc
            else:
                break
        # Assign next color to the group
        if palette_idx < len(palette):
            color = palette[palette_idx]
            palette_idx += 1
            for cc in range(c, group_end + 1):
                for rr in range(run_start, run_end + 1):
                    output[rr][cc] = color
        # Move to next column after group
        c = group_end + 1
    return output