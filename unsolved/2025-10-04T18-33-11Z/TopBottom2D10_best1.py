import numpy as np

def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    grid = np.array(grid_lst)
    height, width = grid.shape
    
    # Find all colored rectangles (non-zero contiguous blocks)
    rectangles = []
    visited = np.zeros_like(grid, dtype=bool)
    
    for i in range(height):
        for j in range(width):
            if grid[i, j] != 0 and not visited[i, j]:
                # Found a new colored rectangle
                color = grid[i, j]
                
                # Find the bounding box of this rectangle
                # First find all connected cells with the same color
                stack = [(i, j)]
                cells = []
                min_row, min_col = i, j
                max_row, max_col = i, j
                
                while stack:
                    r, c = stack.pop()
                    if r < 0 or r >= height or c < 0 or c >= width:
                        continue
                    if visited[r, c] or grid[r, c] != color:
                        continue
                    
                    visited[r, c] = True
                    cells.append((r, c))
                    min_row = min(min_row, r)
                    min_col = min(min_col, c)
                    max_row = max(max_row, r)
                    max_col = max(max_col, c)
                    
                    # Check neighbors
                    for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        stack.append((r + dr, c + dc))
                
                rectangles.append({
                    'color': color,
                    'cells': cells,
                    'min_row': min_row,
                    'max_row': max_row,
                    'min_col': min_col,
                    'max_col': max_col
                })
    
    # Sort rectangles by their vertical position (top to bottom)
    rectangles.sort(key=lambda x: x['min_row'])
    
    # If we have at least 2 rectangles, swap top and bottom
    if len(rectangles) >= 2:
        top_rect = rectangles[0]
        bottom_rect = rectangles[-1]
        
        # Create output grid (copy of input)
        output = grid.copy()
        
        # Clear the areas of top and bottom rectangles
        for r, c in top_rect['cells']:
            output[r, c] = 0
        for r, c in bottom_rect['cells']:
            output[r, c] = 0
        
        # Place bottom rectangle in top position
        row_offset = top_rect['min_row'] - bottom_rect['min_row']
        col_offset = top_rect['min_col'] - bottom_rect['min_col']
        
        for r, c in bottom_rect['cells']:
            new_r = r + row_offset
            new_c = c + col_offset
            if 0 <= new_r < height and 0 <= new_c < width:
                output[new_r, new_c] = bottom_rect['color']
        
        # Place top rectangle in bottom position
        row_offset = bottom_rect['min_row'] - top_rect['min_row']
        col_offset = bottom_rect['min_col'] - top_rect['min_col']
        
        for r, c in top_rect['cells']:
            new_r = r + row_offset
            new_c = c + col_offset
            if 0 <= new_r < height and 0 <= new_c < width:
                output[new_r, new_c] = top_rect['color']
        
        return output.tolist()
    
    # If less than 2 rectangles, return original grid
    return grid.tolist()