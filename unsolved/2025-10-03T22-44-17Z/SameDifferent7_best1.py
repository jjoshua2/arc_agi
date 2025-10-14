import numpy as np

def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    if not grid_lst:
        return []
    grid = np.array(grid_lst)
    rows, cols = grid.shape
    # Find rows with any non-zero
    has_nonzero = np.any(grid != 0, axis=1)
    if not np.any(has_nonzero):
        return grid.tolist()
    r_min = np.min(np.where(has_nonzero))
    
    def get_dominant(row):
        mask = (row != 0) & (row != 8)
        if np.sum(mask) == 0:
            return None
        counts = np.bincount(row[mask], minlength=10)
        return np.argmax(counts)
    
    c = get_dominant(grid[r_min])
    if c is None:
        # No band color, just keep 8's
        result = np.zeros_like(grid)
        result[grid == 8] = 8
        return result.tolist()
    
    r_max = r_min
    while r_max + 1 < rows:
        next_c = get_dominant(grid[r_max + 1])
        if next_c != c:
            break
        r_max += 1
    
    # Set the band to 0
    result = grid.copy()
    result[r_min:r_max + 1, :] = 0
    
    # Set all non-8 to 0 globally
    non_eight_mask = result != 8
    result[non_eight_mask] = 0
    
    return result.tolist()