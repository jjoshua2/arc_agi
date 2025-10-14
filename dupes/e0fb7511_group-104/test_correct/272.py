def find_components(grid):
    rows = len(grid)
    if rows == 0:
        return
    cols = len(grid[0])
    visited = [[False for _ in range(cols)] for _ in range(rows)]
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    components = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 0 and not visited[r][c]:
                comp = []
                stack = [(r, c)]
                visited[r][c] = True
                while stack:
                    cr, cc = stack.pop()
                    comp.append((cr, cc))
                    for dr, dc in directions:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 0 and not visited[nr][nc]:
                            visited[nr][nc] = True
                            stack.append((nr, nc))
                components.append(comp)
    return components

def transform(grid_lst):
    grid = [row[:] for row in grid_lst]
    components = find_components(grid)
    for comp in components:
        if len(comp) >= 2:
            for r, c in comp:
                grid[r][c] = 8
    return grid