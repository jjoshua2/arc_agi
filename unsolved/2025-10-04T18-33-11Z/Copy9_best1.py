import numpy as np

def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    grid = np.array(grid_lst)
    rows, cols = grid.shape
    output = grid.copy().astype(int)
    
    # For each color present
    for color in np.unique(grid):
        if color == 0:
            continue
            
        # Find all cells of this color
        positions = np.argwhere(grid == color)
        n = len(positions)
        
        # For each pair of positions
        for i in range(n):
            for j in range(i+1, n):
                r1, c1 = positions[i]
                r2, c2 = positions[j]
                
                # Check if aligned horizontally (same row)
                if r1 == r2:
                    start_col = min(c1, c2)
                    end_col = max(c1, c2)
                    for c in range(start_col+1, end_col):
                        if output[r1, c] == 0:
                            output[r1, c] = color
                            
                # Check if aligned vertically (same column)
                elif c1 == c2:
                    start_row = min(r1, r2)
                    end_row = max(r1, r2)
                    for r in range(start_row+1, end_row):
                        if output[r, c1] == 0:
                            output[r, c1] = color
                            
                # Check if aligned diagonally (same slope)
                elif abs(r1 - r2) == abs(c1 - c2):
                    step_r = 1 if r2 > r1 else -1
                    step_c = 1 if c2 > c1 else -1
                    r, c = r1 + step_r, c1 + step_c
                    while (r, c) != (r2, c2):
                        if output[r, c] == 0:
                            output[r, c] = color
                        r += step_r
                        c += step_c
    
    return output.tolist()