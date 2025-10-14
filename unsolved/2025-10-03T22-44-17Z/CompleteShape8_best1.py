import numpy as np

def max_consec(arr, val=2):
    arr = np.asarray(arr)
    max_len = 0
    curr_len = 0
    for x in arr:
        if x == val:
            curr_len += 1
            max_len = max(max_len, curr_len)
        else:
            curr_len = 0
    return max_len

def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    if not grid_lst:
        return []
    grid = np.array(grid_lst)
    rows, cols = grid.shape
    
    # Horizontal segments
    h_consec = [max_consec(grid[i, :], 2) for i in range(rows)]
    max_h = max(h_consec) if h_consec else 0
    h_axes = [i for i in range(rows) if h_consec[i] == max_h]
    
    # Vertical segments
    v_consec = [max_consec(grid[:, j], 2) for j in range(cols)]
    max_v = max(v_consec) if v_consec else 0
    v_axes = [j for j in range(cols) if v_consec[j] == max_v]
    
    output = grid.copy()
    
    if max_h > max_v and h_axes:
        # Vertical reflection across horizontal axes
        for i in range(rows):
            for j in range(cols):
                if grid[i, j] == 5:
                    dists = [(abs(i - a), a) for a in h_axes]
                    _, nearest_a = min(dists)
                    new_i = 2 * nearest_a - i
                    if 0 <= new_i < rows:
                        output[new_i, j] = 5
    elif max_v > max_h and v_axes:
        # Horizontal reflection across vertical axes
        for i in range(rows):
            for j in range(cols):
                if grid[i, j] == 5:
                    dists = [(abs(j - b), b) for b in v_axes]
                    _, nearest_b = min(dists)
                    new_j = 2 * nearest_b - j
                    if 0 <= new_j < cols:
                        output[i, new_j] = 5
    # else: no change
    
    # Clear original 5's
    output[grid == 5] = 0
    
    return output.tolist()