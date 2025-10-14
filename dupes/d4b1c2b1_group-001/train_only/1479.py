import numpy as np

def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    grid = np.array(grid_lst)
    rows, cols = grid.shape
    
    # Find all rectangles bounded by blue cells (color 1)
    def find_rectangles():
        rectangles = []
        visited = np.zeros_like(grid, dtype=bool)
        
        for r in range(rows):
            for c in range(cols):
                if grid[r, c] == 1 and not visited[r, c]:
                    # Find the rectangle boundaries
                    # First, find the bottom right corner by scanning right and down
                    r_start, c_start = r, c
                    
                    # Find right boundary
                    c_end = c_start
                    while c_end < cols and grid[r, c_end] == 1:
                        c_end += 1
                    c_end -= 1
                    
                    # Find bottom boundary
                    r_end = r_start
                    while r_end < rows and grid[r_end, c_start] == 1:
                        r_end += 1
                    r_end -= 1
                    
                    # Check if it's a proper rectangle
                    is_rectangle = True
                    for rr in range(r_start, r_end + 1):
                        if grid[rr, c_start] != 1 or grid[rr, c_end] != 1:
                            is_rectangle = False
                            break
                    for cc in range(c_start, c_end + 1):
                        if grid[r_start, cc] != 1 or grid[r_end, cc] != 1:
                            is_rectangle = False
                            break
                    
                    if is_rectangle:
                        rectangles.append((r_start, c_start, r_end, c_end))
                        # Mark all boundary cells as visited
                        for rr in range(r_start, r_end + 1):
                            visited[rr, c_start] = True
                            visited[rr, c_end] = True
                        for cc in range(c_start, c_end + 1):
                            visited[r_start, cc] = True
                            visited[r_end, cc] = True
        
        return rectangles
    
    rectangles = find_rectangles()
    
    # Process each rectangle
    for r_start, c_start, r_end, c_end in rectangles:
        # Check for markers in top row
        for c in range(c_start + 1, c_end):
            if grid[r_start, c] not in [0, 1]:
                marker_color = grid[r_start, c]
                # Fill row above rectangle
                if r_start > 0:
                    for fill_c in range(c_start, c_end + 1):
                        grid[r_start - 1, fill_c] = marker_color
                break
        
        # Check for markers in bottom row
        for c in range(c_start + 1, c_end):
            if grid[r_end, c] not in [0, 1]:
                marker_color = grid[r_end, c]
                # Fill row below rectangle
                if r_end < rows - 1:
                    for fill_c in range(c_start, c_end + 1):
                        grid[r_end + 1, fill_c] = marker_color
                break
    
    return grid.tolist()