import numpy as np

def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    if not grid_lst:
        return []
    grid = np.array(grid_lst)
    rows, cols = grid.shape
    
    # Find bounding box of non-zero cells
    non_zero_pos = np.argwhere(grid > 0)
    if len(non_zero_pos) == 0:
        return grid.tolist()
    
    minr = np.min(non_zero_pos[:, 0])
    maxr = np.max(non_zero_pos[:, 0])
    minc = np.min(non_zero_pos[:, 1])
    maxc = np.max(non_zero_pos[:, 1])
    
    if minr == maxr or minc == maxc:
        return grid.tolist()  # No area, or line
    
    # Frame color from top-left
    frame_color = grid[minr, minc]
    
    inner_minr = minr + 1
    inner_maxr = maxr - 1
    inner_minc = minc + 1
    inner_maxc = maxc - 1
    
    if inner_minr > inner_maxr or inner_minc > inner_maxc:
        return grid.tolist()  # No inner area
    
    # Get the four inner corner colors
    tl_inner = grid[inner_minr, inner_minc]
    tr_inner = grid[inner_minr, inner_maxc]
    bl_inner = grid[inner_maxr, inner_minc]
    br_inner = grid[inner_maxr, inner_maxc]
    
    # Make a copy
    result = grid.copy()
    
    # Clear the inner area
    for r in range(inner_minr, inner_maxr + 1):
        for c in range(inner_minc, inner_maxc + 1):
            result[r, c] = 0
    
    # Place the colors outside if positions are within bounds
    if minr > 0 and minc > 0:
        result[minr - 1, minc - 1] = br_inner  # top-left gets br_inner
    
    if minr > 0 and maxc + 1 < cols:
        result[minr - 1, maxc + 1] = bl_inner  # top-right gets bl_inner
    
    if maxr + 1 < rows and minc > 0:
        result[maxr + 1, minc - 1] = tr_inner  # bottom-left gets tr_inner
    
    if maxr + 1 < rows and maxc + 1 < cols:
        result[maxr + 1, maxc + 1] = tl_inner  # bottom-right gets tl_inner
    
    return result.tolist()