def transform(grid: list[list[int]]) -> list[list[int]]:
    if not grid or not grid[0]:
        return []
    
    h = len(grid)
    w = len(grid[0])
    
    def dfs(start_r: int, start_c: int, color: int) -> int:
        visited = [[False] * w for _ in range(h)]
        stack = [(start_r, start_c)]
        size = 0
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        while stack:
            r, c = stack.pop()
            if visited[r][c]:
                continue
            visited[r][c] = True
            size += 1
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and not visited[nr][nc] and grid[nr][nc] == color:
                    stack.append((nr, nc))
        return size
    
    max_size = 0
    best_color = 0
    
    for i in range(h):
        for j in range(w):
            if grid[i][j] != 0:
                col = grid[i][j]
                # To avoid recounting, we can check if already visited, but since we start only if not visited, wait no.
                # Actually, since we don't have global visited, this will recount components multiple times.
                # Mistake. Need global visited or check if starting a new component.
    
    # Correct way: use a global visited
    visited = [[False] * w for _ in range(h)]
    
    def component_size(r: int, c: int, color: int) -> int:
        if visited[r][c]:
            return 0
        stack = [(r, c)]
        size = 0
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        while stack:
            cr, cc = stack.pop()
            if visited[cr][cc] or grid[cr][cc] != color:
                continue
            visited[cr][cc] = True
            size += 1
            for dr, dc in directions:
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < h and 0 <= nc < w and not visited[nr][nc] and grid[nr][nc] == color:
                    stack.append((nr, nc))
        return size
    
    for i in range(h):
        for j in range(w):
            if grid[i][j] != 0 and not visited[i][j]:
                col = grid[i][j]
                sz = component_size(i, j, col)
                if sz > max_size:
                    max_size = sz
                    best_color = col
    
    # Now create 3x3
    out = [[best_color for _ in range(3)] for _ in range(3)]
    return out