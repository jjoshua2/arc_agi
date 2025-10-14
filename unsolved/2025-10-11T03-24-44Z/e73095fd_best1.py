def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    if not grid_lst or not grid_lst[0]:
        return []
    grid = [row[:] for row in grid_lst]
    rows = len(grid)
    cols = len(grid[0])
    
    # First pass: find the maximum component size of 0s
    visited = [[False for _ in range(cols)] for _ in range(rows)]
    max_size = 0
    
    def compute_size(r, c, vis):
        stack = [(r, c)]
        vis[r][c] = True
        count = 1
        while stack:
            x, y = stack.pop()
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < rows and 0 <= ny < cols and not vis[nx][ny] and grid[nx][ny] == 0:
                    vis[nx][ny] = True
                    stack.append((nx, ny))
                    count += 1
        return count
    
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == 0 and not visited[i][j]:
                comp_size = compute_size(i, j, visited)
                if comp_size > max_size:
                    max_size = comp_size
    
    # Second pass: fill components that are not the largest
    visited = [[False for _ in range(cols)] for _ in range(rows)]
    
    def process_component(r, c, vis):
        positions = []
        stack = [(r, c)]
        vis[r][c] = True
        positions.append((r, c))
        while stack:
            x, y = stack.pop()
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < rows and 0 <= ny < cols and not vis[nx][ny] and grid[nx][ny] == 0:
                    vis[nx][ny] = True
                    stack.append((nx, ny))
                    positions.append((nx, ny))
        comp_size = len(positions)
        if comp_size != max_size:
            for px, py in positions:
                grid[px][py] = 4
    
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == 0 and not visited[i][j]:
                process_component(i, j, visited)
    
    return grid