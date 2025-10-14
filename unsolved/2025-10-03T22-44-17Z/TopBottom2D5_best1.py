import numpy as np

def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    grid = np.array(grid_lst)
    if grid.size == 0:
        return grid_lst

    # Find non-zero colors and pick the most frequent as the main color
    vals, counts = np.unique(grid, return_counts=True)
    nonzero = vals[vals != 0]
    if nonzero.size == 0:
        return grid.tolist()
    # Choose the most frequent non-zero color as main color
    # (occluder(s) will be removed)
    main_color = nonzero[np.argmax([counts[np.where(vals == v)[0][0]] for v in nonzero])]

    # Mask of main color
    m_mask = (grid == main_color)
    if not m_mask.any():
        # No main color found; return zeros of same shape
        return np.zeros_like(grid).tolist()

    # Bounding box of main color
    coords = np.argwhere(m_mask)
    rmin, cmin = coords.min(axis=0)
    rmax, cmax = coords.max(axis=0)

    # Prepare output as all zeros (occluders removed by default)
    out = np.zeros_like(grid)

    # Region of interest (ROI) for symmetry
    roi = m_mask[rmin:rmax+1, cmin:cmax+1]
    # Mirror horizontally within the ROI
    roi_mirror = roi[:, ::-1]
    # Union to enforce horizontal symmetry
    roi_sym = np.logical_or(roi, roi_mirror)

    # Fill ROI with main color where symmetric mask is true
    out[rmin:rmax+1, cmin:cmax+1][roi_sym] = main_color

    # Return as list of lists
    return out.tolist()