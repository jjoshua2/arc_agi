import numpy as np

def transform(grid: list[list[int]]) -> list[list[int]]:
    grid = np.array(grid)
    rows, cols = grid.shape
    
    # Find the green bars (3) to determine sections
    # Green bars are at columns 5, 10, 15, 20 (0-indexed)
    # So sections are:
    # Section 1: columns 0-4
    # Section 2: columns 5-9
    # Section 3: columns 10-14
    # Section 4: columns 15-19
    # Column 20 is the right border
    
    # Create output grid
    output = grid.copy()
    
    # First, set all yellow cells to 0
    output[output == 4] = 0
    
    # Then, for each yellow cell in the input, place it in the new position
    for r in range(rows):
        for c in range(cols):
            if grid[r, c] == 4:
                if 0 <= c <= 4:  # Section 1
                    new_c = c + 15
                    if new_c < cols:  # Within bounds
                        output[r, new_c] = 4
                elif 15 <= c <= 19:  # Section 4
                    if r > 0:  # Can move up
                        output[r-1, c] = 4
                    # else, disappears (already set to 0)
                else:  # Other sections
                    output[r, c] = 4
    
    return output.tolist()