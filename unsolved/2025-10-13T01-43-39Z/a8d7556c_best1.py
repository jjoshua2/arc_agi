import numpy as np

def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    if not grid_lst or not grid_lst[0]:
        return []
    
    grid = np.array(grid_lst)
    rows, cols = grid.shape
    binary = (grid == 0).astype(int)
    
    # Compute prefix sums
    prefix = np.zeros((rows + 1, cols + 1), dtype=int)
    for i in range(1, rows + 1):
        for j in range(1, cols + 1):
            prefix[i, j] = prefix[i - 1, j] + prefix[i, j - 1] - prefix[i - 1, j - 1] + binary[i - 1, j - 1]
    
    def get_sum(t, b, l, r):
        return prefix[b + 1, r + 1] - prefix[b + 1, l] - prefix[t, r + 1] + prefix[t, l]
    
    output = grid.copy()
    
    for t in range(rows):
        for b in range(t + 1, rows):  # height = b - t + 1 >= 2
            for l in range(cols):
                for r in range(l + 1, cols):  # width = r - l + 1 >= 2
                    height = b - t + 1
                    width = r - l + 1
                    if get_sum(t, b, l, r) == height * width:
                        # Check maximality
                        can_extend_up = t > 0 and get_sum(t - 1, t - 1, l, r) == width
                        can_extend_down = b < rows - 1 and get_sum(b + 1, b + 1, l, r) == width
                        can_extend_left = l > 0 and get_sum(t, b, l - 1, l - 1) == height
                        can_extend_right = r < cols - 1 and get_sum(t, b, r + 1, r + 1) == height
                        if not (can_extend_up or can_extend_down or can_extend_left or can_extend_right):
                            output[t:b + 1, l:r + 1] = 2
    
    return output.tolist()