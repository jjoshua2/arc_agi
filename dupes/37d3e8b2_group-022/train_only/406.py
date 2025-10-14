from collections import deque

def transform(grid: list[list[int]]) -> list[list[int]]:
    if not grid or not grid[0]:
        return []
    rows = len(grid)
    cols = len(grid[0])
    grid_out = [row[:] for row in grid]
    visited = [[False] * cols for _ in range(rows)]
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    components = []
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == 8 and not visited[i][j]:
                comp = []
                stack = [(i, j)]
                visited[i][j] = True
                while stack:
                    x, y = stack.pop()
                    comp.append((x, y))
                    for dx, dy in directions:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < rows and 0 <= ny < cols and grid[nx][ny] == 8 and not visited[nx][ny]:
                            visited[nx][ny] = True
                            stack.append((nx, ny))
                if comp:
                    components.append(comp)
    for comp in components:
        comp_set = set(comp)
        candidates = set()
        for x, y in comp:
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < rows and 0 <= ny < cols and grid[nx][ny] == 0:
                    candidates.add((nx, ny))
        hole_visited = [[False] * cols for _ in range(rows)]
        num_holes = 0
        for sx, sy in candidates:
            if hole_visited[sx][sy]:
                continue
            q = deque([(sx, sy)])
            hole_visited[sx][sy] = True
            is_hole = True
            if sx == 0 or sx == rows - 1 or sy == 0 or sy == cols - 1:
                is_hole = False
            while q:
                ux, uy = q.popleft()
                for dx, dy in directions:
                    nx, ny = ux + dx, uy + dy
                    if 0 <= nx < rows and 0 <= ny < cols:
                        if grid[nx][ny] == 0:
                            if not hole_visited[nx][ny]:
                                hole_visited[nx][ny] = True
                                q.append((nx, ny))
                                if nx == 0 or nx == rows - 1 or ny == 0 or ny == cols - 1:
                                    is_hole = False
                        elif grid[nx][ny] == 8 and (nx, ny) not in comp_set:
                            is_hole = False
            if is_hole:
                num_holes += 1
        color = num_holes
        for x, y in comp:
            grid_out[x][y] = color
    return grid_out