from collections import deque

def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    if not grid_lst or not grid_lst[0]:
        return grid_lst
    
    rows, cols = len(grid_lst), len(grid_lst[0])
    output = [row[:] for row in grid_lst]  # Create a copy
    
    # Find all connected components using 8-connectivity
    visited = [[False] * cols for _ in range(rows)]
    components = []
    
    for r in range(rows):
        for c in range(cols):
            if not visited[r][c] and output[r][c] != 0:
                color = output[r][c]
                queue = deque([(r, c)])
                component = []
                min_r, max_r = r, r
                min_c, max_c = c, c
                
                visited[r][c] = True
                component.append((r, c))
                
                while queue:
                    cr, cc = queue.popleft()
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            if dr == 0 and dc == 0:
                                continue
                            nr, nc = cr + dr, cc + dc
                            if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc] and output[nr][nc] == color:
                                visited[nr][nc] = True
                                component.append((nr, nc))
                                queue.append((nr, nc))
                                min_r = min(min_r, nr)
                                max_r = max(max_r, nr)
                                min_c = min(min_c, nc)
                                max_c = max(max_c, nc)
                
                components.append({
                    'color': color,
                    'cells': component,
                    'min_r': min_r,
                    'max_r': max_r,
                    'min_c': min_c,
                    'max_c': max_c,
                    'size': len(component)
                })
    
    if not components:
        return output
    
    # Find the largest component
    largest_component = max(components, key=lambda x: x['size'])
    
    # Change top row of the bounding box to color 5
    top_row = largest_component['min_r']
    for c in range(largest_component['min_c'], largest_component['max_c'] + 1):
        if output[top_row][c] == largest_component['color']:
            output[top_row][c] = 5
    
    return output