import numpy as np
from collections import deque

def transform(grid_lst):
    if not grid_lst or not grid_lst[0]:
        return grid_lst
    
    grid = np.array(grid_lst)
    rows, cols = grid.shape
    visited = np.zeros_like(grid, dtype=bool)
    purple = 8
    
    # Directions for 4-connectivity
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    # Find all purple rectangles
    rectangles = []
    
    for r in range(rows):
        for c in range(cols):
            if grid[r, c] == purple and not visited[r, c]:
                # BFS to find connected purple component
                queue = deque([(r, c)])
                visited[r, c] = True
                min_r, max_r = r, r
                min_c, max_c = c, c
                cells = [(r, c)]
                
                while queue:
                    x, y = queue.popleft()
                    for dx, dy in directions:
                        nx, ny = x + dx, y + dy
                        if (0 <= nx < rows and 0 <= ny < cols and 
                            grid[nx, ny] == purple and not visited[nx, ny]):
                            visited[nx, ny] = True
                            queue.append((nx, ny))
                            cells.append((nx, ny))
                            min_r = min(min_r, nx)
                            max_r = max(max_r, nx)
                            min_c = min(min_c, ny)
                            max_c = max(max_c, ny)
                
                rectangles.append({
                    'cells': cells,
                    'min_r': min_r,
                    'min_c': min_c
                })
    
    # Sort rectangles in reading order (top to bottom, left to right)
    rectangles.sort(key=lambda x: (x['min_r'], x['min_c']))
    
    # Assign colors in reverse order (first gets highest number, last gets lowest)
    # We'll use colors 1, 2, 3, 4, ... based on position
    color_map = {}
    for i, rect in enumerate(rectangles):
        # Assign color: first rectangle gets len(rectangles), last gets 1
        color = len(rectangles) - i
        for cell in rect['cells']:
            color_map[cell] = color
    
    # Create output grid
    output = grid.copy()
    for (r, c), color in color_map.items():
        output[r, c] = color
    
    return output.tolist()