import numpy as np

def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    grid = np.array(grid_lst)
    h, w = grid.shape
    
    # Create output grid with doubled dimensions
    out_h, out_w = 2 * h, 2 * w
    output = np.zeros((out_h, out_w), dtype=int)
    
    # Create the 8 pattern (checkerboard)
    pattern = np.zeros((out_h, out_w), dtype=int)
    for i in range(out_h):
        for j in range(out_w):
            if (i % 2 == 0 and j % 2 == 0) or (i % 2 == 1 and j % 2 == 1):
                pattern[i, j] = 8
    
    # Tile the input pattern
    for i in range(2):
        for j in range(2):
            output[i*h:(i+1)*h, j*w:(j+1)*w] = grid
    
    # Overlay the 8 pattern, but preserve non-zero values from input
    for i in range(out_h):
        for j in range(out_w):
            if output[i, j] == 0:
                output[i, j] = pattern[i, j]
    
    return output.tolist()