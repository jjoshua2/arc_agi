def transform(grid: list[list[int]]) -> list[list[int]]:
    if not grid or not grid[0]:
        return []
    rows = len(grid)
    cols = len(grid[0])
    visited = [[False] * cols for _ in range(rows)]
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    candidates = []

    for r in range(rows):
        for c in range(cols):
            if not visited[r][c] and grid[r][c] != 0 and grid[r][c] != 5:
                # Start DFS
                stack = [(r, c)]
                visited[r][c] = True
                min_r = max_r = r
                min_c = max_c = c
                comp_size = 0
                max_color = grid[r][c]
                while stack:
                    cr, cc = stack.pop()
                    comp_size += 1
                    min_r = min(min_r, cr)
                    max_r = max(max_r, cr)
                    min_c = min(min_c, cc)
                    max_c = max(max_c, cc)
                    max_color = max(max_color, grid[cr][cc])
                    for dr, dc in directions:
                        nr = cr + dr
                        nc = cc + dc
                        if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc] and grid[nr][nc] != 0 and grid[nr][nc] != 5:
                            visited[nr][nc] = True
                            stack.append((nr, nc))
                h = max_r - min_r + 1
                w = max_c - min_c + 1
                if h * w == comp_size:
                    candidates.append((comp_size, max_color, min_r, min_c, max_r, max_c))

    if not candidates:
        return []

    # Select the best: max size, then min max_color, then min min_r, then min min_c
    best = max(candidates, key=lambda t: (t[0], -t[1], -t[2], -t[3]))
    _, _, min_r, min_c, max_r, max_c = best
    h = max_r - min_r + 1
    w = max_c - min_c + 1
    out = []
    for i in range(h):
        row = [grid[min_r + i][min_c + j] for j in range(w)]
        out.append(row)
    return out