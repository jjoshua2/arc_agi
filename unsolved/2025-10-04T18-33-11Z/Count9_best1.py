from collections import deque

def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    if not grid_lst or not grid_lst[0]:
        return [[]]
    
    rows = len(grid_lst)
    cols = len(grid_lst[0])
    
    # Directions for 4-connectivity
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    visited = [[False for _ in range(cols)] for _ in range(rows)]
    component_count = 0
    
    for r in range(rows):
        for c in range(cols):
            if grid_lst[r][c] != 0 and not visited[r][c]:
                component_count += 1
                queue = deque([(r, c)])
                visited[r][c] = True
                
                while queue:
                    curr_r, curr_c = queue.popleft()
                    
                    for dr, dc in directions:
                        nr, nc = curr_r + dr, curr_c + dc
                        if (0 <= nr < rows and 0 <= nc < cols and 
                            not visited[nr][nc] and grid_lst[nr][nc] != 0):
                            visited[nr][nc] = True
                            queue.append((nr, nc))
    
    # Return a row with component_count zeros
    return [[0] * component_count]