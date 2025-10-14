def transform(grid: list[list[int]]) -> list[list[int]]:
    if not grid:
        return []
    h = len(grid)
    w = len(grid[0])
    output = [row[:] for row in grid]
    visited = [[False] * w for _ in range(h)]
    directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    for i in range(h):
        for j in range(w):
            if grid[i][j] == 8 and not visited[i][j]:
                # new component
                component = []
                stack = [(i, j)]
                visited[i][j] = True
                while stack:
                    r, c = stack.pop()
                    component.append((r, c))
                    for dr, dc in directions:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < h and 0 <= nc < w and not visited[nr][nc] and grid[nr][nc] == 8:
                            visited[nr][nc] = True
                            stack.append((nr, nc))
                if component:
                    min_r = min(r for r, c in component)
                    max_r = max(r for r, c in component)
                    min_c = min(c for r, c in component)
                    max_c = max(c for r, c in component)
                    for rr in range(min_r, max_r + 1):
                        for cc in range(min_c, max_c + 1):
                            if output[rr][cc] == 0:
                                output[rr][cc] = 2
    return output