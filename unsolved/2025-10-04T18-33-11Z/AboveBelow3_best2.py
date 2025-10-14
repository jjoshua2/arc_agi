import numpy as np

def transform(grid_lst):
    if not grid_lst or not grid_lst[0]:
        return grid_lst
    
    grid = np.array(grid_lst)
    rows, cols = grid.shape
    
    # Find connected components (8-connectivity)
    visited = np.zeros_like(grid, dtype=bool)
    components = []
    directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    
    for r in range(rows):
        for c in range(cols):
            if grid[r, c] != 0 and not visited[r, c]:
                # New component
                stack = [(r, c)]
                visited[r, c] = True
                positions = []
                min_row, max_row = r, r
                min_col, max_col = c, c
                
                while stack:
                    x, y = stack.pop()
                    positions.append((x, y))
                    min_row = min(min_row, x)
                    max_row = max(max_row, x)
                    min_col = min(min_col, y)
                    max_col = max(max_col, y)
                    
                    for dx, dy in directions:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < rows and 0 <= ny < cols:
                            if grid[nx, ny] == grid[r, c] and not visited[nx, ny]:
                                visited[nx, ny] = True
                                stack.append((nx, ny))
                
                components.append({
                    'color': grid[r, c],
                    'positions': positions,
                    'min_row': min_row,
                    'max_row': max_row,
                    'min_col': min_col,
                    'max_col': max_col
                })
    
    # Create a new grid initially all zeros
    new_grid = np.zeros_like(grid)
    
    # For each component, calculate how far it can fall
    for comp in components:
        fall_distance = 0
        current_bottom = comp['max_row']
        col_start, col_end = comp['min_col'], comp['max_col']
        
        # Check how many empty rows are below the component
        for fall_row in range(current_bottom + 1, rows):
            # Check if all cells in the column range are empty in this row
            if np.all(grid[fall_row, col_start:col_end+1] == 0):
                fall_distance += 1
            else:
                break
        
        # Shift the component downward by fall_distance
        new_positions = [(r + fall_distance, c) for r, c in comp['positions']]
        for (r, c) in new_positions:
            if 0 <= r < rows and 0 <= c < cols:
                new_grid[r, c] = comp['color']
    
    # Remove empty rows from the top
    non_empty_rows = np.any(new_grid != 0, axis=1)
    first_non_empty = np.argmax(non_empty_rows)
    last_non_empty = len(non_empty_rows) - np.argmax(non_empty_rows[::-1]) - 1
    
    # If there are empty rows at the top, remove them
    if first_non_empty > 0:
        new_grid = new_grid[first_non_empty:last_non_empty+1, :]
    
    return new_grid.tolist()