import numpy as np

def get_connected_components(grid):
    """Get all connected components of non-zero colors, returning list of (color, component_size, representative_pos)"""
    rows, cols = grid.shape
    visited = np.zeros((rows, cols), dtype=bool)
    components = []
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    for r in range(rows):
        for c in range(cols):
            if grid[r, c] != 0 and not visited[r, c]:
                color = grid[r, c]
                component_size = 0
                stack = [(r, c)]
                visited[r, c] = True
                
                while stack:
                    cr, cc = stack.pop()
                    component_size += 1
                    
                    for dr, dc in directions:
                        nr, nc = cr + dr, cc + dc
                        if (0 <= nr < rows and 0 <= nc < cols and 
                            grid[nr, nc] == color and not visited[nr, nc]):
                            visited[nr, nc] = True
                            stack.append((nr, nc))
                
                components.append((color, component_size, (r, c)))
    
    return components

def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    if not grid_lst or not grid_lst[0]:
        return []
    
    grid = np.array(grid_lst)
    rows, cols = grid.shape
    
    components = get_connected_components(grid)
    
    if not components:
        # All zero, perhaps return 3x3 zero, but assume not
        return [[0]*3 for _ in range(3)]
    
    # Find the component with max size
    max_comp = max(components, key=lambda x: x[1])
    color = max_comp[0]
    
    # Create 3x3 output
    output = [[color for _ in range(3)] for _ in range(3)]
    return output