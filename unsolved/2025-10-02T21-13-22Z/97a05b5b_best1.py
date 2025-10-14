import numpy as np

def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    grid = np.array(grid_lst)
    rows, cols = grid.shape
    
    # Find the main shape (non-zero region)
    visited = np.zeros_like(grid, dtype=bool)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    # Find all connected components
    components = []
    for i in range(rows):
        for j in range(cols):
            if grid[i, j] != 0 and not visited[i, j]:
                stack = [(i, j)]
                visited[i, j] = True
                component = []
                min_i, max_i = i, i
                min_j, max_j = j, j
                
                while stack:
                    x, y = stack.pop()
                    component.append((x, y))
                    min_i = min(min_i, x)
                    max_i = max(max_i, x)
                    min_j = min(min_j, y)
                    max_j = max(max_j, y)
                    
                    for dx, dy in directions:
                        nx, ny = x + dx, y + dy
                        if (0 <= nx < rows and 0 <= ny < cols and 
                            not visited[nx, ny] and grid[nx, ny] != 0):
                            visited[nx, ny] = True
                            stack.append((nx, ny))
                
                components.append((component, min_i, max_i, min_j, max_j))
    
    # Find the largest component (main shape)
    if not components:
        return []
    
    main_component = max(components, key=lambda x: len(x[0]))
    main_cells, main_min_i, main_max_i, main_min_j, main_max_j = main_component
    
    # Create output grid with main shape filled with 2s
    out_rows = main_max_i - main_min_i + 1
    out_cols = main_max_j - main_min_j + 1
    output = np.full((out_rows, out_cols), 2, dtype=int)
    
    # Find all other components (patterns)
    patterns = []
    for comp, min_i, max_i, min_j, max_j in components:
        if comp != main_cells:
            pattern = grid[min_i:max_i+1, min_j:max_j+1]
            # Calculate relative position to main shape
            rel_i = min_i - main_min_i
            rel_j = min_j - main_min_j
            patterns.append((pattern, rel_i, rel_j))
    
    # Insert patterns into output
    for pattern, rel_i, rel_j in patterns:
        p_rows, p_cols = pattern.shape
        for i in range(p_rows):
            for j in range(p_cols):
                if (0 <= rel_i + i < out_rows and 
                    0 <= rel_j + j < out_cols):
                    output[rel_i + i, rel_j + j] = pattern[i, j]
    
    return output.tolist()