def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    if not grid_lst:
        return []
    grid = [row[:] for row in grid_lst]
    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0
    visited = [[False] * cols for _ in range(rows)]
    components = []
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == 2 and not visited[i][j]:
                color = 2
                stack = [(i, j)]
                pos = []
                min_c_this = j
                while stack:
                    r, c = stack.pop()
                    if visited[r][c]:
                        continue
                    visited[r][c] = True
                    pos.append((r, c))
                    min_c_this = min(min_c_this, c)
                    for dr, dc in directions:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc] and grid[nr][nc] == color:
                            stack.append((nr, nc))
                if pos:
                    components.append((len(pos), color, min_c_this, pos))
    if not components:
        return grid_lst
    max_size = max(s[0] for s in components)
    candidates = [c for c in components if c[0] == max_size]
    best = min(candidates, key=lambda x: x[2])
    red_pos = best[3]
    min_r = min(r for r, c in red_pos)
    min_c = min(c for r, c in red_pos)
    relatives = [(r - min_r, c - min_c) for r, c in red_pos]
    max_dr = max(dr for dr, dc in relatives)
    max_dc = max(dc for dr, dc in relatives)
    b_height = max_dr + 1
    b_width = max_dc + 1
    if b_height == 1:
        fixed_ar = min_r
        for ac in range(cols - b_width + 1):
            if all(0 <= ac + dc < cols and grid[fixed_ar + dr][ac + dc] == 0 for dr, dc in relatives):
                for dr, dc in relatives:
                    grid[fixed_ar + dr][ac + dc] = 2
    elif b_width == 1:
        fixed_ac = min_c
        for ar in range(rows - b_height + 1):
            if all(0 <= ar + dr < rows and grid[ar + dr][fixed_ac + dc] == 0 for dr, dc in relatives):
                for dr, dc in relatives:
                    grid[ar + dr][fixed_ac + dc] = 2
    else:
        for ar in range(rows - b_height + 1):
            for ac in range(cols - b_width + 1):
                if all(0 <= ar + dr < rows and 0 <= ac + dc < cols and grid[ar + dr][ac + dc] == 0 for dr, dc in relatives):
                    for dr, dc in relatives:
                        grid[ar + dr][ac + dc] = 2
    return grid