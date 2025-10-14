import numpy as np
from collections import deque

def transform(grid_lst):
    grid = np.array(grid_lst)
    h, w = grid.shape
    output = grid.copy()
    
    # Find all connected components
    visited = np.zeros_like(grid, dtype=bool)
    components = []
    
    for i in range(h):
        for j in range(w):
            if grid[i, j] != 0 and not visited[i, j]:
                color = grid[i, j]
                component = []
                queue = deque([(i, j)])
                visited[i, j] = True
                
                while queue:
                    x, y = queue.popleft()
                    component.append((x, y))
                    
                    for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < h and 0 <= ny < w and not visited[nx, ny] and grid[nx, ny] == color:
                            visited[nx, ny] = True
                            queue.append((nx, ny))
                
                components.append((color, component))
    
    # For each component, apply transformation based on color and shape
    for color, cells in components:
        if color == 4:  # Yellow
            # Check if this is adjacent to blue (1) cells
            blue_adjacent = False
            for x, y in cells:
                for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < h and 0 <= ny < w and grid[nx, ny] == 1:
                        blue_adjacent = True
                        break
                if blue_adjacent:
                    break
            
            if blue_adjacent:
                # Expand to form a plus shape around blue cells
                for x, y in cells:
                    # Expand in all four directions
                    for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < h and 0 <= ny < w and output[nx, ny] == 0:
                            output[nx, ny] = 4
        
        elif color == 6:  # Pink
            # Expand vertically to form a line
            min_x = min(x for x, y in cells)
            max_x = max(x for x, y in cells)
            min_y = min(y for x, y in cells)
            max_y = max(y for x, y in cells)
            
            if max_x - min_x == 0:  # Vertical line
                for y in range(min_y, max_y + 1):
                    for dx in [-1, 1]:
                        nx = min_x + dx
                        if 0 <= nx < h and output[nx, y] == 0:
                            output[nx, y] = 6
        
        elif color == 3:  # Green
            # Expand horizontally to form a line or cross
            min_x = min(x for x, y in cells)
            max_x = max(x for x, y in cells)
            min_y = min(y for x, y in cells)
            max_y = max(y for x, y in cells)
            
            if max_y - min_y == 0:  # Horizontal line
                for x in range(min_x, max_x + 1):
                    for dy in [-1, 1]:
                        ny = min_y + dy
                        if 0 <= ny < w and output[x, ny] == 0:
                            output[x, ny] = 3
            else:
                # Form a cross - expand in both directions
                center_x = (min_x + max_x) // 2
                center_y = (min_y + max_y) // 2
                
                for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    nx, ny = center_x + dx, center_y + dy
                    if 0 <= nx < h and 0 <= ny < w and output[nx, ny] == 0:
                        output[nx, ny] = 3
        
        elif color == 8:  # Purple
            # Expand to form a frame around other colored blocks
            min_x = min(x for x, y in cells)
            max_x = max(x for x, y in cells)
            min_y = min(y for x, y in cells)
            max_y = max(y for x, y in cells)
            
            # Check what's inside the purple block
            inner_colors = set()
            for x in range(min_x, max_x + 1):
                for y in range(min_y, max_y + 1):
                    if grid[x, y] != 0 and grid[x, y] != 8:
                        inner_colors.add(grid[x, y])
            
            if inner_colors:
                # Expand to form a frame around the inner colors
                frame_size = 3  # Default frame size
                
                # Expand the frame
                for x in range(min_x - frame_size, max_x + frame_size + 1):
                    for y in range(min_y - frame_size, max_y + frame_size + 1):
                        if (x < min_x or x > max_x or y < min_y or y > max_y) and \
                           0 <= x < h and 0 <= y < w and output[x, y] == 0:
                            output[x, y] = 8
    
    return output.tolist()