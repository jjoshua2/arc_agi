def transform(grid: list[list[int]]) -> list[list[int]]:
    if not grid or not grid[0]:
        return []
    rows = len(grid)
    cols = len(grid[0])
    visited = [[False] * cols for _ in range(rows)]
    components = []
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == 2 and not visited[i][j]:
                comp = []
                stack = [(i, j)]
                visited[i][j] = True
                while stack:
                    x, y = stack.pop()
                    comp.append((x, y))
                    for dx, dy in directions:
                        nx = x + dx
                        ny = y + dy
                        if 0 <= nx < rows and 0 <= ny < cols and grid[nx][ny] == 2 and not visited[nx][ny]:
                            visited[nx][ny] = True
                            stack.append((nx, ny))
                components.append(comp)
    if not components:
        return [row[:] for row in grid]
    # Compute center_c for each component
    comp_with_center = []
    for comp in components:
        total_c = sum(c for _, c in comp)
        center_c = total_c / len(comp)
        comp_with_center.append((center_c, comp))
    # Sort by center_c
    comp_with_center.sort(key=lambda x: x[0])
    N = len(comp_with_center)
    output = [row[:] for row in grid]
    for i in range(N):
        if (i % 2) != (N % 2):
            for r, c in comp_with_center[i][1]:
                output[r][c] = 8
    return output