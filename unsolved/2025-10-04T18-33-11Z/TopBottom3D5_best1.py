import numpy as np
from collections import deque

def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    grid = np.array(grid_lst)
    rows, cols = grid.shape
    
    if rows == 0 or cols == 0:
        return []
    
    # Directions for 4-connected components
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    visited = np.zeros_like(grid, dtype=bool)
    components = []
    
    # Find all connected components of non-zero cells
    for i in range(rows):
        for j in range(cols):
            if grid[i, j] != 0 and not visited[i, j]:
                color = grid[i, j]
                component = []
                queue = deque([(i, j)])
                visited[i, j] = True
                
                min_r, max_r = i, i
                min_c, max_c = j, j
                
                while queue:
                    x, y = queue.popleft()
                    component.append((x, y))
                    min_r = min(min_r, x)
                    max_r = max(max_r, x)
                    min_c = min(min_c, y)
                    max_c = max(max_c, y)
                    
                    for dx, dy in directions:
                        nx, ny = x + dx, y + dy
                        if (0 <= nx < rows and 0 <= ny < cols and 
                            not visited[nx, ny] and grid[nx, ny] == color):
                            visited[nx, ny] = True
                            queue.append((nx, ny))
                
                components.append((color, component, min_r, max_r, min_c, max_c))
    
    if not components:
        return [[0]]
    
    # Find the largest component by area
    largest_component = max(components, key=lambda x: len(x[1]))
    color, positions, min_r, max_r, min_c, max_c = largest_component
    
    # Calculate bounding box dimensions
    height = max_r - min_r + 1
    width = max_c - min_c + 1
    
    # Determine output grid size based on bounding box aspect ratio
    # Use a fixed smaller size while preserving aspect ratio
    max_size = 4  # Based on examples, output is at most 4 in any dimension
    
    if width > height:
        out_width = min(width, max_size)
        out_height = max(1, int(height * out_width / width))
    else:
        out_height = min(height, max_size)
        out_width = max(1, int(width * out_height / height))
    
    # Create output grid
    output_grid = [[0] * out_width for _ in range(out_height)]
    
    # Scale the pattern
    for r, c in positions:
        # Map original position to output position
        out_r = int((r - min_r) * out_height / height)
        out_c = int((c - min_c) * out_width / width)
        
        # Ensure within bounds
        if 0 <= out_r < out_height and 0 <= out_c < out_width:
            output_grid[out_r][out_c] = color
    
    return output_grid