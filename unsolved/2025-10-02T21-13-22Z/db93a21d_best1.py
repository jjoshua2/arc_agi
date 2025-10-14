import numpy as np
from collections import deque

def transform(grid_lst):
    grid = np.array(grid_lst)
    rows, cols = grid.shape
    
    # Find all brown (9) rectangles
    rectangles = []
    visited = np.zeros_like(grid, dtype=bool)
    
    for i in range(rows):
        for j in range(cols):
            if grid[i, j] == 9 and not visited[i, j]:
                # Find the bounding box of this rectangle
                queue = deque([(i, j)])
                visited[i, j] = True
                points = [(i, j)]
                
                while queue:
                    x, y = queue.popleft()
                    for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < rows and 0 <= ny < cols and not visited[nx, ny] and grid[nx, ny] == 9:
                            visited[nx, ny] = True
                            queue.append((nx, ny))
                            points.append((nx, ny))
                
                if points:
                    min_i = min(p[0] for p in points)
                    max_i = max(p[0] for p in points)
                    min_j = min(p[1] for p in points)
                    max_j = max(p[1] for p in points)
                    rectangles.append((min_i, max_i, min_j, max_j))
    
    # Create output grid
    output = grid.copy()
    
    # For each rectangle, create green border
    for min_i, max_i, min_j, max_j in rectangles:
        # Expand the rectangle by 1 in each direction for green border
        border_min_i = max(0, min_i - 1)
        border_max_i = min(rows - 1, max_i + 1)
        border_min_j = max(0, min_j - 1)
        border_max_j = min(cols - 1, max_j + 1)
        
        # Fill green border
        for i in range(border_min_i, border_max_i + 1):
            for j in range(border_min_j, border_max_j + 1):
                if output[i, j] == 0:
                    output[i, j] = 3
        
        # Create blue border inside green border
        blue_min_i = max(0, min_i)
        blue_max_i = min(rows - 1, max_i)
        blue_min_j = max(0, min_j)
        blue_max_j = min(cols - 1, max_j)
        
        for i in range(blue_min_i, blue_max_i + 1):
            for j in range(blue_min_j, blue_max_j + 1):
                if output[i, j] == 0:
                    output[i, j] = 1
    
    return output.tolist()