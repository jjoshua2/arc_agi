from collections import deque

def transform(grid: list[list[int]]) -> list[list[int]]:
    if not grid or not grid[0]:
        return grid
    
    rows = len(grid)
    cols = len(grid[0])
    output = [row[:] for row in grid]
    
    # Directions for horizontal and vertical movement
    # We'll process horizontal and vertical extensions separately
    processed_h = set()  # tracks (row, col) that have been processed for horizontal extension
    processed_v = set()  # tracks (col, row) that have been processed for vertical extension
    queue = deque()
    
    # Initialize queue with all existing red cells (color 2)
    for i in range(rows):
        for j in range(cols):
            if output[i][j] == 2:
                queue.append(('h', i, j))
    
    while queue:
        task_type, param1, param2 = queue.popleft()
        
        if task_type == 'h':
            # Horizontal extension from cell (param1, param2)
            r, c = param1, param2
            
            if (r, c) in processed_h:
                continue
            processed_h.add((r, c))
            
            # Extend right
            current_col = c
            while True:
                next_col = current_col + 1
                if next_col >= cols:
                    break
                if output[r][next_col] == 0:
                    output[r][next_col] = 2
                    current_col = next_col
                elif output[r][next_col] == 2:
                    current_col = next_col
                else:
                    # Hit a colored cell (non-black, non-red)
                    queue.append(('v', current_col, r))
                    break
            
            # Extend left
            current_col = c
            while True:
                next_col = current_col - 1
                if next_col < 0:
                    break
                if output[r][next_col] == 0:
                    output[r][next_col] = 2
                    current_col = next_col
                elif output[r][next_col] == 2:
                    current_col = next_col
                else:
                    # Hit a colored cell (non-black, non-red)
                    queue.append(('v', current_col, r))
                    break
        
        elif task_type == 'v':
            # Vertical extension from column param1, row param2
            c, r = param1, param2
            
            if (c, r) in processed_v:
                continue
            processed_v.add((c, r))
            
            # Extend down
            current_row = r
            while True:
                next_row = current_row + 1
                if next_row >= rows:
                    break
                if output[next_row][c] == 0:
                    output[next_row][c] = 2
                    current_row = next_row
                elif output[next_row][c] == 2:
                    current_row = next_row
                else:
                    # Hit a colored cell (non-black, non-red)
                    queue.append(('h', current_row, c))
                    break
            
            # Extend up
            current_row = r
            while True:
                next_row = current_row - 1
                if next_row < 0:
                    break
                if output[next_row][c] == 0:
                    output[next_row][c] = 2
                    current_row = next_row
                elif output[next_row][c] == 2:
                    current_row = next_row
                else:
                    # Hit a colored cell (non-black, non-red)
                    queue.append(('h', current_row, c))
                    break
    
    return output