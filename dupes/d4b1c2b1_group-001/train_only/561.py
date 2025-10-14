import numpy as np
from collections import deque

def transform(grid_lst):
    grid = np.array(grid_lst)
    h, w = grid.shape
    output = grid.copy()
    
    # Find all connected components
    visited = np.zeros_like(grid, dtype=bool)
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    
    def is_valid(r, c):
        return 0 <= r < h and 0 <= c < w
    
    def bfs(r, c, color):
        queue = deque([(r, c)])
        component = []
        while queue:
            cr, cc = queue.popleft()
            if visited[cr, cc]:
                continue
            visited[cr, cc] = True
            component.append((cr, cc))
            for dr, dc in directions:
                nr, nc = cr + dr, cc + dc
                if is_valid(nr, nc) and not visited[nr, nc] and grid[nr, nc] == color:
                    queue.append((nr, nc))
        return component
    
    # Find all components
    components = []
    for r in range(h):
        for c in range(w):
            if grid[r, c] != 0 and not visited[r, c]:
                color = grid[r, c]
                component = bfs(r, c, color)
                components.append((color, component))
    
    # For each component, check if it's a rectangle with hollow interior
    for color, component in components:
        if len(component) < 4:  # Too small to be a rectangle
            continue
            
        # Find bounding box
        rows = [p[0] for p in component]
        cols = [p[1] for p in component]
        min_r, max_r = min(rows), max(rows)
        min_c, max_c = min(cols), max(cols)
        
        # Check if it forms a rectangle
        expected_cells = (max_r - min_r + 1) * (max_c - min_c + 1)
        if len(component) != expected_cells:
            continue  # Not a perfect rectangle
            
        # Check if it's hollow (has interior zeros)
        has_interior_zeros = False
        for r in range(min_r + 1, max_r):
            for c in range(min_c + 1, max_c):
                if grid[r, c] == 0:
                    has_interior_zeros = True
                    break
            if has_interior_zeros:
                break
                
        if not has_interior_zeros:
            continue  # Not hollow
            
        # This is a hollow rectangle, apply maze transformation
        # Create a maze in the interior
        interior_height = max_r - min_r - 1
        interior_width = max_c - min_c - 1
        
        if interior_height <= 0 or interior_width <= 0:
            continue
            
        # Create a maze using a simple algorithm
        # We'll use a depth-first search approach
        maze = np.zeros((interior_height, interior_width), dtype=int)
        
        # Mark borders as walls
        maze[0, :] = 1
        maze[-1, :] = 1
        maze[:, 0] = 1
        maze[:, -1] = 1
        
        # Use DFS to create paths
        stack = [(1, 1)]
        visited_maze = np.zeros_like(maze, dtype=bool)
        
        while stack:
            r, c = stack.pop()
            if visited_maze[r, c]:
                continue
            visited_maze[r, c] = True
            maze[r, c] = 1  # Mark as path
            
            # Shuffle directions for randomness
            dirs = directions.copy()
            np.random.shuffle(dirs)
            
            for dr, dc in dirs:
                nr, nc = r + dr, c + dc
                if 0 <= nr < interior_height and 0 <= nc < interior_width and not visited_maze[nr, nc]:
                    # Check if we can move there (not too close to border)
                    if 1 <= nr < interior_height - 1 and 1 <= nc < interior_width - 1:
                        stack.append((nr, nc))
        
        # Copy the maze to the output
        for r in range(interior_height):
            for c in range(interior_width):
                if maze[r, c] == 1:
                    output[min_r + 1 + r, min_c + 1 + c] = color
    
    return output.tolist()