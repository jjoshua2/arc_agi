import numpy as np

def count_components(grid):
    if grid.size == 0:
        return 0
    rows, cols = grid.shape
    visited = np.zeros((rows, cols), dtype=bool)
    count = 0
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    for r in range(rows):
        for c in range(cols):
            if grid[r, c] == 8 and not visited[r, c]:
                count += 1
                stack = [(r, c)]
                visited[r, c] = True
                
                while stack:
                    cr, cc = stack.pop()
                    
                    for dr, dc in directions:
                        nr, nc = cr + dr, cc + dc
                        if (0 <= nr < rows and 0 <= nc < cols and
                            not visited[nr, nc] and grid[nr, nc] == 8):
                            visited[nr, nc] = True
                            stack.append((nr, nc))
    
    return count

def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    if not grid_lst or not grid_lst[0]:
        return []
    
    grid = np.array(grid_lst)
    n = count_components(grid)
    if n == 0:
        return []
    
    output = np.zeros((n, n), dtype=int)
    np.fill_diagonal(output, 8)
    return output.tolist()