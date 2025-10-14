def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    if not grid_lst or not grid_lst[0]:
        return [row[:] for row in grid_lst]
    
    h = len(grid_lst)
    w = len(grid_lst[0])
    
    visited = [[False] * w for _ in range(h)]
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    components = []
    
    for i in range(h):
        for j in range(w):
            if grid_lst[i][j] == 0 and not visited[i][j]:
                stack = [(i, j)]
                visited[i][j] = True
                comp_size = 1
                comp_cells = [(i, j)]
                
                while stack:
                    x, y = stack.pop()
                    for dx, dy in directions:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < h and 0 <= ny < w and grid_lst[nx][ny] == 0 and not visited[nx][ny]:
                            visited[nx][ny] = True
                            stack.append((nx, ny))
                            comp_size += 1
                            comp_cells.append((nx, ny))
                
                components.append((comp_size, comp_cells))
    
    if not components:
        return [row[:] for row in grid_lst]
    
    max_size = max(size for size, _ in components)
    
    output = [row[:] for row in grid_lst]
    for size, cells in components:
        if size < max_size:
            for x, y in cells:
                output[x][y] = 4
    
    return output