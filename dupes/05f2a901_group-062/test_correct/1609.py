import numpy as np
from collections import deque

def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    grid = np.array(grid_lst)
    rows, cols = grid.shape
    
    # Create output grid
    output = np.zeros_like(grid)
    
    # Find all purple rectangles (color 8)
    purple_rectangles = []
    visited = np.zeros_like(grid, dtype=bool)
    
    for r in range(rows):
        for c in range(cols):
            if grid[r, c] == 8 and not visited[r, c]:
                # Find the rectangle boundaries
                left = c
                right = c
                while right < cols and grid[r, right] == 8 and not visited[r, right]:
                    right += 1
                right -= 1
                
                bottom = r
                while bottom < rows and all(grid[bottom, col] == 8 and not visited[bottom, col] for col in range(left, right + 1)):
                    bottom += 1
                bottom -= 1
                
                # Mark as visited and add to rectangles
                for rr in range(r, bottom + 1):
                    for cc in range(left, right + 1):
                        visited[rr, cc] = True
                
                purple_rectangles.append((r, bottom, left, right))
    
    # Find all colored shapes (non-zero, non-purple)
    colored_shapes = []
    visited = np.zeros_like(grid, dtype=bool)
    
    for r in range(rows):
        for c in range(cols):
            if grid[r, c] != 0 and grid[r, c] != 8 and not visited[r, c]:
                # Flood fill to find the shape
                shape_cells = []
                color = grid[r, c]
                queue = deque([(r, c)])
                visited[r, c] = True
                
                while queue:
                    cr, cc = queue.popleft()
                    shape_cells.append((cr, cc))
                    
                    for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                        nr, nc = cr + dr, cc + dc
                        if (0 <= nr < rows and 0 <= nc < cols and 
                            grid[nr, nc] == color and not visited[nr, nc]):
                            visited[nr, nc] = True
                            queue.append((nr, nc))
                
                colored_shapes.append((color, shape_cells))
    
    # For each colored shape, find the closest purple rectangle and move toward it
    for color, shape_cells in colored_shapes:
        if not shape_cells:
            continue
            
        # Find bounding box of the shape
        shape_rows = [r for r, c in shape_cells]
        shape_cols = [c for r, c in shape_cells]
        min_r, max_r = min(shape_rows), max(shape_rows)
        min_c, max_c = min(shape_cols), max(shape_cols)
        
        # Find closest purple rectangle
        closest_rect = None
        min_distance = float('inf')
        
        for rect_r, rect_b, rect_l, rect_rr in purple_rectangles:
            # Calculate distance between shape and rectangle
            # Use center points for distance calculation
            shape_center_r = (min_r + max_r) / 2
            shape_center_c = (min_c + max_c) / 2
            rect_center_r = (rect_r + rect_b) / 2
            rect_center_c = (rect_l + rect_rr) / 2
            
            distance = ((shape_center_r - rect_center_r) ** 2 + 
                        (shape_center_c - rect_center_c) ** 2) ** 0.5
            
            if distance < min_distance:
                min_distance = distance
                closest_rect = (rect_r, rect_b, rect_l, rect_rr)
        
        if closest_rect is None:
            # No purple rectangle found, keep shape in place
            for r, c in shape_cells:
                output[r, c] = color
            continue
        
        rect_r, rect_b, rect_l, rect_rr = closest_rect
        
        # Determine movement direction
        shape_above = max_r < rect_r
        shape_below = min_r > rect_b
        shape_left = max_c < rect_l
        shape_right = min_c > rect_rr
        
        # Calculate movement vector
        dr, dc = 0, 0
        
        if shape_above:
            # Move down until adjacent
            dr = rect_r - max_r - 1
        elif shape_below:
            # Move up until adjacent
            dr = rect_b - min_r + 1
        elif shape_left:
            # Move right until adjacent
            dc = rect_l - max_c - 1
        elif shape_right:
            # Move left until adjacent
            dc = rect_rr - min_c + 1
        
        # Move the shape
        for r, c in shape_cells:
            new_r = r + dr
            new_c = c + dc
            if 0 <= new_r < rows and 0 <= new_c < cols:
                output[new_r, new_c] = color
    
    # Keep purple rectangles in place
    for rect_r, rect_b, rect_l, rect_rr in purple_rectangles:
        for r in range(rect_r, rect_b + 1):
            for c in range(rect_l, rect_rr + 1):
                output[r, c] = 8
    
    return output.tolist()