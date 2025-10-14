import numpy as np

def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    grid_arr = np.array(grid_lst)
    rows, cols = grid_arr.shape
    
    # Find width w: number of consecutive 8's from left in row 0
    w = 0
    for j in range(cols):
        if grid_arr[0, j] == 8:
            w += 1
        else:
            break
    
    if w == 0:
        return grid_lst
    
    # Count frequencies in row 8
    row8 = grid_arr[8, :]
    freq = {}
    for c in row8:
        if c != 0:
            freq[c] = freq.get(c, 0) + 1
    
    # The empty area rows: 1 to 5, but we fill from bottom
    empty_end = 5
    fill_start_row = empty_end - w + 1  # 0-indexed
    
    # For each color c with freq[c] == w
    for c, f in freq.items():
        if f == w and c != 8:
            # Find columns where row8 == c
            positions = np.where(row8 == c)[0]
            for j in positions:
                # Fill from fill_start_row to 5 in col j with c
                grid_arr[fill_start_row:empty_end+1, j] = c
    
    return grid_arr.tolist()