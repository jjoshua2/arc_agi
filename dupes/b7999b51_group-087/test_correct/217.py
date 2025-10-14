def transform(grid: list[list[int]]) -> list[list[int]]:
    def get_components(grid):
        if not grid or not grid[0]:
            return []
        h = len(grid)
        w = len(grid[0])
        visited = [[False] * w for _ in range(h)]
        components = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for r in range(h):
            for c in range(w):
                if grid[r][c] != 0 and not visited[r][c]:
                    color = grid[r][c]
                    stack = [(r, c)]
                    visited[r][c] = True
                    min_r_ = r
                    max_r_ = r
                    min_c_ = c
                    while stack:
                        cr, cc = stack.pop()
                        min_r_ = min(min_r_, cr)
                        max_r_ = max(max_r_, cr)
                        min_c_ = min(min_c_, cc)
                        for dr, dc in directions:
                            nr, nc = cr + dr, cc + dc
                            if 0 <= nr < h and 0 <= nc < w and not visited[nr][nc] and grid[nr][nc] == color:
                                visited[nr][nc] = True
                                stack.append((nr, nc))
                    height = max_r_ - min_r_ + 1
                    components.append((height, min_c_, min_r_, color))
        return components

    comps = get_components(grid)
    if not comps:
        return []
    comps.sort(key=lambda x: (-x[0], x[1], x[2]))
    max_h = max(c[0] for c in comps)
    width = len(comps)
    output = [[0] * width for _ in range(max_h)]
    for j in range(width):
        height, _, _, col = comps[j]
        for i in range(height):
            output[i][j] = col
    return output