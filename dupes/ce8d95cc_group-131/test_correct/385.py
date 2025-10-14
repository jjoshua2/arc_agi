import numpy as np

def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    if not grid_lst or not grid_lst[0]:
        return []
    grid = np.array(grid_lst, dtype=int)
    rows, cols = grid.shape

    # Compute row modes
    row_modes = []
    for r in range(rows):
        counts = np.bincount(grid[r], minlength=10)
        mode = np.argmax(counts)
        row_modes.append(mode)
    row_modes = np.array(row_modes)

    # Find horizontal band starts and lengths
    row_change_indices = np.where(row_modes[:-1] != row_modes[1:])[0] + 1
    row_band_starts = np.concatenate(([0], row_change_indices))
    row_band_ends = np.concatenate((row_change_indices, [rows]))
    row_band_lengths = row_band_ends - row_band_starts

    # Compute column modes
    grid_t = grid.T
    col_modes = []
    for c in range(cols):
        counts = np.bincount(grid_t[c], minlength=10)
        mode = np.argmax(counts)
        col_modes.append(mode)
    col_modes = np.array(col_modes)

    # Find vertical band starts and lengths
    col_change_indices = np.where(col_modes[:-1] != col_modes[1:])[0] + 1
    col_band_starts = np.concatenate(([0], col_change_indices))
    col_band_ends = np.concatenate((col_change_indices, [cols]))
    col_band_lengths = col_band_ends - col_band_starts

    H = len(row_band_lengths)
    V = len(col_band_lengths)

    # Cumulative starts for rows and columns
    row_starts = np.concatenate(([0], np.cumsum(row_band_lengths[:-1])))
    col_starts = np.concatenate(([0], np.cumsum(col_band_lengths[:-1])))

    # Create output
    output = np.zeros((H, V), dtype=int)
    for i in range(H):
        r_start = int(row_starts[i])
        r_end = r_start + row_band_lengths[i]
        for j in range(V):
            c_start = int(col_starts[j])
            c_end = c_start + col_band_lengths[j]
            subgrid = grid[r_start:r_end, c_start:c_end]
            sub_flat = subgrid.flatten()
            if len(sub_flat) > 0:
                counts = np.bincount(sub_flat, minlength=10)
                output[i, j] = np.argmax(counts)

    return output.tolist()