import numpy as np

def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    grid = np.array(grid_lst)
    rows, cols = grid.shape
    
    # Create output as copy of input
    result = grid.copy()
    
    # Find all potential rectangular regions
    for r1 in range(rows):
        for c1 in range(cols):
            # Skip if current cell is black (0)
            if grid[r1, c1] == 0:
                continue
                
            # Try to find the bottom-right corner of a rectangle starting at (r1, c1)
            border_color = grid[r1, c1]
            
            # Find the maximum rectangle with this border color
            r2 = r1
            c2 = c1
            
            # Expand right
            while c2 + 1 < cols and grid[r1, c2 + 1] == border_color:
                c2 += 1
                
            # Expand down
            while r2 + 1 < rows and grid[r2 + 1, c1] == border_color:
                r2 += 1
                
            # Check if this forms a valid rectangle with solid border
            if r2 <= r1 or c2 <= c1:
                continue
                
            # Check if all border cells have the same color
            valid = True
            # Check top and bottom borders
            for c in range(c1, c2 + 1):
                if grid[r1, c] != border_color or grid[r2, c] != border_color:
                    valid = False
                    break
                    
            # Check left and right borders
            if valid:
                for r in range(r1, r2 + 1):
                    if grid[r, c1] != border_color or grid[r, c2] != border_color:
                        valid = False
                        break
            
            if valid:
                # This is a valid bordered rectangle
                # Set all interior cells to 0
                for r in range(r1 + 1, r2):
                    for c in range(c1 + 1, c2):
                        result[r, c] = 0
    
    return result.tolist()