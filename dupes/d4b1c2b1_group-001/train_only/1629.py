import numpy as np

def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    grid = np.array(grid_lst)
    h, w = grid.shape
    
    # Create output grid as copy of input
    output = grid.copy()
    
    # Directions: right, down, left, up
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    
    # Find pairs of adjacent colored cells (non-zero, non-8)
    for r in range(h):
        for c in range(w):
            color = grid[r, c]
            if color != 0 and color != 8:  # Colored cell that's not purple
                # Check all four directions for adjacent same-colored cells
                for dr, dc in directions:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] == color:
                        # Found a pair, determine fill direction
                        fill_dr, fill_dc = dr, dc
                        
                        # Start filling in the direction away from the pair
                        fill_r, fill_c = r + fill_dr, c + fill_dc
                        while (0 <= fill_r < h and 0 <= fill_c < w and 
                               output[fill_r, fill_c] == 0):
                            output[fill_r, fill_c] = color
                            fill_r += fill_dr
                            fill_c += fill_dc
    
    return output.tolist()