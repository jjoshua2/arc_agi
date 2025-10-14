import numpy as np

def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    if not grid_lst or not grid_lst[0]:
        return []
    grid = np.array(grid_lst)
    rows, cols = grid.shape

    # Dominant color per row
    row_modes = []
    for r in range(rows):
        counts = np.bincount(grid[r], minlength=10)
        mode = np.argmax(counts)
        row_modes.append(mode)

    # Horizontal bands: groups of consecutive rows with same mode
    horizontal_bands = []
    if row_modes:
        start = 0
        for r in range(1, rows):
            if row_modes[r] != row_modes[r - 1]:
                horizontal_bands.append((start, r - 1))
                start = r
        horizontal_bands.append((start, rows - 1))

    # Dominant color per column
    col_modes = []
    for c in range(cols):
        counts = np.bincount(grid[:, c], minlength=10)
        mode = np.argmax(counts)
        col_modes.append(mode)

    # Vertical bands: groups of consecutive columns with same mode
    vertical_bands = []
    if col_modes:
        start = 0
        for c in range(1, cols):
            if col_modes[c] != col_modes[c - 1]:
                vertical_bands.append((start, c - 1))
                start = c
        vertical_bands.append((start, cols - 1))

    # Output dimensions
    out_height = len(horizontal_bands)
    out_width = len(vertical_bands)
    if out_height == 0 or out_width == 0:
        return []

    out = np.zeros((out_height, out_width), dtype=int)

    # Fill output with dominant color of each band intersection
    for i, (r_start, r_end) in enumerate(horizontal_bands):
        for j, (c_start, c_end) in enumerate(vertical_bands):
            subgrid = grid[r_start:r_end + 1, c_start:c_end + 1]
            flat = subgrid.flatten()
            counts = np.bincount(flat, minlength=10)
            mode = np.argmax(counts)
            out[i, j] = mode

    return out.tolist()