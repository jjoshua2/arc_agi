import numpy as np
from collections import deque

def transform(grid_lst):
    if not grid_lst or not grid_lst[0]:
        return grid_lst
    
    grid = np.array(grid_lst)
    h, w = grid.shape
    
    # Create visited matrix
    visited = np.zeros_like(grid, dtype=bool)
    
    # Directions for 4-connectivity
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    def bfs(r, c):
        """BFS to find connected component and its bounding box"""
        color = grid[r, c]
        component = []
        queue = deque([(r, c)])
        visited[r, c] = True
        min_r, max_r = r, r
        min_c, max_c = c, c
        
        while queue:
            x, y = queue.popleft()
            component.append((x, y))
            min_r = min(min_r, x)
            max_r = max(max_r, x)
            min_c = min(min_c, y)
            max_c = max(max_c, y)
            
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if (0 <= nx < h and 0 <= ny < w and not visited[nx, ny] 
                    and grid[nx, ny] == color):
                    visited[nx, ny] = True
                    queue.append((nx, ny))
        
        return component, (min_r, max_r, min_c, max_c)
    
    # Find all connected components
    components = []
    for i in range(h):
        for j in range(w):
            if grid[i, j] != 0 and not visited[i, j]:
                comp, bbox = bfs(i, j)
                components.append((comp, bbox, grid[i, j]))
    
    # Sort components by size (largest first)
    components.sort(key=lambda x: len(x[0]), reverse=True)
    
    # Create output grid (start with all zeros)
    output = np.zeros_like(grid)
    
    # Keep only the largest components (shapes)
    for comp, bbox, color in components:
        if len(comp) > 10:  # Threshold for considering it a shape
            for r, c in comp:
                output[r, c] = color
    
    # Now fill holes in the shapes based on nearby colors
    # For each cell in output that's 0 but surrounded by non-zero cells
    for i in range(1, h-1):
        for j in range(1, w-1):
            if output[i, j] == 0:
                # Check if it's surrounded by non-zero cells (a hole)
                neighbors = []
                for dx, dy in directions:
                    ni, nj = i + dx, j + dy
                    if output[ni, nj] != 0:
                        neighbors.append(output[ni, nj])
                
                if len(neighbors) >= 3:  # Mostly surrounded
                    # Find the most common color around this hole
                    if neighbors:
                        # Use the color from the original grid if available
                        if grid[i, j] != 0:
                            output[i, j] = grid[i, j]
                        else:
                            # Use the most common neighbor color
                            from collections import Counter
                            counter = Counter(neighbors)
                            most_common = counter.most_common(1)
                            if most_common:
                                output[i, j] = most_common[0][0]
    
    return output.tolist()