import numpy as np

def transform(grid: list[list[int]]) -> list[list[int]]:
    grid_arr = np.array(grid)
    rows, cols = grid_arr.shape
    
    # Find width w: number of consecutive 8's from left in row 0
    w = 0
    for j in range(cols):
        if grid_arr[0, j] == 8:
            w += 1
        else:
            break
    
    if w == 0:
        return grid
    
    # Count frequencies in row 8
    row8 = grid_arr[8, :]
    freq = {}
    for c in row8:
        if c != 0:
            freq[c] = freq.get(c, 0) + 1
    
    # The empty area rows: 1 to 5
    empty_start = 1
    empty_end = 5
    num_empty = empty_end - empty_start + 1  # 5
    
    # For each color c with freq[c] == w
    fill_start_row = empty_end - w + 1  # e.g., for w=3, 5-3+1=3 (0-indexed row 3)
    
    for c, f in freq.items():
        if f == w and c != 8:  # assuming not filling with 8, as per examples
            # Find columns where row8 == c
            positions = np.where(row8 == c)[0]
            for j in positions:
                # Fill from fill_start_row to 5 in col j with c
                grid_arr[fill_start_row:empty_end+1, j] = c
    
    return grid_arr.tolist()