import numpy as np
import math

def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    grid = np.array(grid_lst, dtype=int)
    rows, cols = grid.shape

    # thresholds (require the dominant non-zero color to cover this fraction to be a clear axis)
    frac = 0.6
    row_thresh = max(1, math.ceil(cols * frac))
    col_thresh = max(1, math.ceil(rows * frac))

    # helper to get dominant non-zero color count for a 1D array
    def dominant_nonzero(arr):
        nonzeros = arr[arr != 0]
        if nonzeros.size == 0:
            return (0, 0)
        vals, cnts = np.unique(nonzeros, return_counts=True)
        idx = int(np.argmax(cnts))
        return (int(vals[idx]), int(cnts[idx]))

    # collect row scores (row_index, color, count)
    row_scores = []
    for r in range(rows):
        color, count = dominant_nonzero(grid[r, :])
        row_scores.append((r, color, count))

    # collect column scores (col_index, color, count)
    col_scores = []
    for c in range(cols):
        color, count = dominant_nonzero(grid[:, c])
        col_scores.append((c, color, count))

    # find candidate rows meeting threshold
    row_candidates = [t for t in row_scores if t[2] >= row_thresh and t[1] != 0]
    if len(row_candidates) < 2:
        # fallback: pick top-2 rows by dominant non-zero count (ignore rows with count 0 if possible)
        sorted_rows = sorted(row_scores, key=lambda x: x[2], reverse=True)
        # ensure unique indices and ignore zero-count rows if possible
        chosen = []
        for t in sorted_rows:
            if t[2] > 0:
                chosen.append(t)
            if len(chosen) >= 2:
                break
        # if still <2, fill with top rows regardless
        if len(chosen) < 2:
            chosen = sorted_rows[:2]
        row_candidates = chosen

    # find candidate cols meeting threshold
    col_candidates = [t for t in col_scores if t[2] >= col_thresh and t[1] != 0]
    if len(col_candidates) < 2:
        # fallback: pick top-2 cols by dominant non-zero count
        sorted_cols = sorted(col_scores, key=lambda x: x[2], reverse=True)
        chosen = []
        for t in sorted_cols:
            if t[2] > 0:
                chosen.append(t)
            if len(chosen) >= 2:
                break
        if len(chosen) < 2:
            chosen = sorted_cols[:2]
        col_candidates = chosen

    # get the two row indices and two col indices (take extremes)
    row_indices = sorted([t[0] for t in row_candidates])[:2]
    if len(row_indices) > 2:
        row_indices = row_indices[:2]
    # If there are more than two candidates (rare), pick the top two by count then sort by index
    if len(row_indices) < 2:
        # fallback take top two from row_scores
        sorted_rows = sorted(row_scores, key=lambda x: x[2], reverse=True)
        row_indices = sorted([sorted_rows[0][0], sorted_rows[1][0]])

    col_indices = sorted([t[0] for t in col_candidates])[:2]
    if len(col_indices) > 2:
        col_indices = col_indices[:2]
    if len(col_indices) < 2:
        sorted_cols = sorted(col_scores, key=lambda x: x[2], reverse=True)
        col_indices = sorted([sorted_cols[0][0], sorted_cols[1][0]])

    r0, r1 = min(row_indices), max(row_indices)
    c0, c1 = min(col_indices), max(col_indices)

    # Crop inclusive
    cropped = grid[r0:r1+1, c0:c1+1].tolist()
    return cropped