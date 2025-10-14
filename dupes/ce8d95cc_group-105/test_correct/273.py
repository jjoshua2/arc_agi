def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    if not grid_lst or not grid_lst[0]:
        return []
    grid = [row[:] for row in grid_lst]
    h = len(grid)
    w = len(grid[0])
    # Compute nz_col
    nz_col = [sum(1 for r in range(h) if grid[r][c] != 0) for c in range(w)]
    # Vertical cols: those with nz_col[c] > h // 2
    vertical_cols = [c for c in range(w) if nz_col[c] > h // 2]
    vertical_cols.sort()
    num_v = len(vertical_cols)
    v_set = set(vertical_cols)
    # Now num_v known
    # Compute nz_row
    nz_row = [sum(1 for c in range(w) if grid[r][c] != 0) for r in range(h)]
    # Horizontal rows: nz_row[r] > num_v
    horizontal_rows = [r for r in range(h) if nz_row[r] > num_v]
    horizontal_rows.sort()
    num_h = len(horizontal_rows)
    h_set = set(horizontal_rows)
    # Now, vertical_colors
    vertical_color = []
    non_h_row = next((r for r in range(h) if r not in h_set), None)
    if non_h_row is not None:
        for v_c in vertical_cols:
            vertical_color.append(grid[non_h_row][v_c])
    else:
        # If no non_h, perhaps use 0 or something, but assume exists
        vertical_color = [0] * num_v
    # Now, output dimensions
    output_h = 2 * num_h + 1 if num_h > 0 else 1
    output_w = 2 * num_v + 1 if num_v > 0 else 1
    output_grid = [[0 for _ in range(output_w)] for _ in range(output_h)]
    # Now fill
    for out_r in range(output_h):
        if out_r % 2 == 0:  # empty
            for out_c in range(output_w):
                if out_c % 2 == 0:  # gap
                    output_grid[out_r][out_c] = 0
                else:  # vertical
                    v_idx = (out_c - 1) // 2
                    output_grid[out_r][out_c] = vertical_color[v_idx]
        else:  # horizontal
            h_idx = (out_r - 1) // 2
            in_r = horizontal_rows[h_idx]
            # Find h_color: find a non-vertical col
            h_color = 0
            for c in range(w):
                if c not in v_set and grid[in_r][c] != 0:
                    h_color = grid[in_r][c]
                    break
            # Assume found
            for out_c in range(output_w):
                if out_c % 2 == 0:  # gap
                    output_grid[out_r][out_c] = h_color
                else:  # vertical
                    v_idx = (out_c - 1) // 2
                    v_c = vertical_cols[v_idx]
                    output_grid[out_r][out_c] = grid[in_r][v_c]
    return output_grid