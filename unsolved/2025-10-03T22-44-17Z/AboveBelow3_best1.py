import numpy as np

def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    grid = np.array(grid_lst)
    output = grid.copy()
    
    # Find all non-zero components
    def find_components(arr):
        rows, cols = arr.shape
        visited = np.zeros_like(arr, dtype=bool)
        components = []
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
        for r in range(rows):
            for c in range(cols):
                if arr[r, c] != 0 and not visited[r, c]:
                    color = arr[r, c]
                    stack = [(r, c)]
                    component = []
                    visited[r, c] = True
                    
                    while stack:
                        cr, cc = stack.pop()
                        component.append((cr, cc))
                        for dr, dc in directions:
                            nr, nc = cr + dr, cc + dc
                            if (0 <= nr < rows and 0 <= nc < cols and 
                                not visited[nr, nc] and arr[nr, nc] == color):
                                visited[nr, nc] = True
                                stack.append((nr, nc))
                    
                    components.append({
                        'color': color,
                        'positions': component,
                        'min_r': min(p[0] for p in component),
                        'max_r': max(p[0] for p in component),
                        'min_c': min(p[1] for p in component),
                        'max_c': max(p[1] for p in component)
                    })
        return components
    
    components = find_components(grid)
    
    if len(components) < 2:
        return output.tolist()
    
    # Find the largest component (the one that stays in place)
    largest_component = max(components, key=lambda c: len(c['positions']))
    
    # Find the smaller component to move (not the largest one)
    smaller_components = [c for c in components if c != largest_component]
    if not smaller_components:
        return output.tolist()
    
    # For simplicity, take the first smaller component
    # In real implementation, might need more complex logic to choose the right one
    moving_component = smaller_components[0]
    
    # Determine movement direction based on relative position
    l_min_r, l_max_r = largest_component['min_r'], largest_component['max_r']
    l_min_c, l_max_c = largest_component['min_c'], largest_component['max_c']
    m_min_r, m_max_r = moving_component['min_r'], moving_component['max_r']
    m_min_c, m_max_c = moving_component['min_c'], moving_component['max_c']
    
    # Clear the moving component from its original position
    for r, c in moving_component['positions']:
        output[r, c] = 0
    
    # Determine target position based on relative position
    if m_min_r > l_max_r:  # Moving component is below
        # Move up to align with bottom of largest component
        target_r = l_max_r + 1
        target_c = l_min_c
        # Create a horizontal bar at the bottom
        for c in range(l_min_c, l_max_c + 1):
            output[target_r, c] = moving_component['color']
    elif m_max_r < l_min_r:  # Moving component is above
        # Move down to align with top of largest component
        target_r = l_min_r - 1
        target_c = l_min_c
        # Create a horizontal bar at the top
        for c in range(l_min_c, l_max_c + 1):
            output[target_r, c] = moving_component['color']
    elif m_min_c > l_max_c:  # Moving component is to the right
        # Move left to align with right side of largest component
        target_r = l_min_r
        target_c = l_max_c + 1
        # Create a vertical bar on the right
        for r in range(l_min_r, l_max_r + 1):
            output[r, target_c] = moving_component['color']
    elif m_max_c < l_min_c:  # Moving component is to the left
        # Move right to align with left side of largest component
        target_r = l_min_r
        target_c = l_min_c - 1
        # Create a vertical bar on the left
        for r in range(l_min_r, l_max_r + 1):
            output[r, target_c] = moving_component['color']
    
    return output.tolist()