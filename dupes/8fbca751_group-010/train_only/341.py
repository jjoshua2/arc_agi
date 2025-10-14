def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    if not grid_lst or not grid_lst[0]:
        return []
    rows = len(grid_lst)
    cols = len(grid_lst[0])
    grid = [row[:] for row in grid_lst]
    visited = [[False for _ in range(cols)] for _ in range(rows)]
    
    dirs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    
    def dfs(r, c, component):
        stack = [(r, c)]
        visited[r][c] = True
        component.append((r, c))
        while stack:
            cr, cc = stack.pop()
            for dr, dc in dirs:
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc] and grid[nr][nc] == 8:
                    visited[nr][nc] = True
                    component.append((nr, nc))
                    stack.append((nr, nc))
    
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == 8 and not visited[i][j]:
                component = []
                dfs(i, j, component)
                if component:
                    rs = [pos[0] for pos in component]
                    cs = [pos[1] for pos in component]
                    min_r = min(rs)
                    max_r = max(rs)
                    min_c = min(cs)
                    max_c = max(cs)
                    for rr in range(min_r, max_r + 1):
                        for cc in range(min_c, max_c + 1):
                            if grid[rr][cc] == 0:
                                grid[rr][cc] = 2
    return grid