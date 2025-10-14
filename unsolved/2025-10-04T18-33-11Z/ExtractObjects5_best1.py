import numpy as np
from collections import defaultdict

def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    grid = np.array(grid_lst)
    h, w = grid.shape
    
    # Find all connected components of non-zero values
    visited = np.zeros_like(grid, dtype=bool)
    components = []
    
    def dfs(r, c, component):
        if r < 0 or r >= h or c < 0 or c >= w or visited[r, c] or grid[r, c] == 0:
            return
        visited[r, c] = True
        component.append((r, c, grid[r, c]))
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            dfs(r + dr, c + dc, component)
    
    for r in range(h):
        for c in range(w):
            if grid[r, c] != 0 and not visited[r, c]:
                component = []
                dfs(r, c, component)
                components.append(component)
    
    # Extract bounding boxes and patterns for each component
    shapes = []
    for component in components:
        if not component:
            continue
            
        rows = [p[0] for p in component]
        cols = [p[1] for p in component]
        min_r, max_r = min(rows), max(rows)
        min_c, max_c = min(cols), max(cols)
        
        # Create pattern matrix
        pattern_h = max_r - min_r + 1
        pattern_w = max_c - min_c + 1
        pattern = np.zeros((pattern_h, pattern_w), dtype=int)
        
        for r, c, val in component:
            pattern[r - min_r, c - min_c] = val
        
        shapes.append(pattern)
    
    # Remove duplicate shapes (based on pattern content)
    unique_shapes = []
    seen_patterns = set()
    for shape in shapes:
        pattern_str = str(shape.tolist())
        if pattern_str not in seen_patterns:
            seen_patterns.add(pattern_str)
            unique_shapes.append(shape)
    
    if not unique_shapes:
        return [[]]
    
    # Determine output dimensions
    max_height = max(shape.shape[0] for shape in unique_shapes)
    total_width = sum(shape.shape[1] for shape in unique_shapes)
    
    # Create output grid
    output = np.zeros((max_height, total_width), dtype=int)
    
    current_col = 0
    for shape in unique_shapes:
        shape_h, shape_w = shape.shape
        # Center the shape vertically
        start_row = (max_height - shape_h) // 2
        output[start_row:start_row + shape_h, current_col:current_col + shape_w] = shape
        current_col += shape_w
    
    return output.tolist()