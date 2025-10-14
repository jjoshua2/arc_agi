def transform(input_grid: list[list[int]]) -> list[list[int]]:
    if not input_grid or not input_grid[0]:
        return input_grid
    
    rows = len(input_grid)
    cols = len(input_grid[0])
    grid = [row[:] for row in input_grid]
    visited = [[False for _ in range(cols)] for _ in range(rows)]
    
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right
    
    def dfs(r, c, component):
        stack = [(r, c)]
        visited[r][c] = True
        component.append((r, c))
        
        while stack:
            cr, cc = stack.pop()
            for dr, dc in directions:
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc] and grid[nr][nc] == 0:
                    visited[nr][nc] = True
                    component.append((nr, nc))
                    stack.append((nr, nc))
    
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == 0 and not visited[i][j]:
                component = []
                dfs(i, j, component)
                
                # Check if component touches border
                touches_border = False
                for cr, cc in component:
                    if cr == 0 or cr == rows - 1 or cc == 0 or cc == cols - 1:
                        touches_border = True
                        break
                
                # If not touching border, fill with 3
                if not touches_border:
                    for cr, cc in component:
                        grid[cr][cc] = 3
    
    return grid