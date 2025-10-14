import numpy as np

def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    grid = np.array(grid_lst)
    output = grid.copy()
    
    # Find the vertical separation (column of zeros that divides the grid)
    cols = grid.shape[1]
    sep_col = None
    for c in range(cols):
        if np.all(grid[:, c] == 0):
            sep_col = c
            break
            
    if sep_col is None:
        return output.tolist()
    
    # Determine left and right regions
    left_region = grid[:, :sep_col]
    right_region = grid[:, sep_col+1:]
    
    # Find the background color of left region (most frequent non-zero color)
    left_nonzero = left_region[left_region != 0]
    if len(left_nonzero) == 0:
        return output.tolist()
    background_color = np.bincount(left_nonzero).argmax()
    
    # Mirror the non-background, non-zero cells from left to right
    left_width = left_region.shape[1]
    right_width = right_region.shape[1]
    
    for r in range(left_region.shape[0]):
        for c in range(left_region.shape[1]):
            color = left_region[r, c]
            if color != 0 and color != background_color:
                # Mirror position in right region
                mirror_c = right_width - 1 - c
                if mirror_c >= 0 and mirror_c < right_width:
                    output[r, sep_col+1+mirror_c] = color
                    
    return output.tolist()