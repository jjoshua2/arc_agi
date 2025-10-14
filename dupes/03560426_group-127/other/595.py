def transform(grid: list[list[int]]) -> list[list[int]]:
    H = 10
    W = 10
    visited = [[False] * W for _ in range(H)]
    components = []
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for i in range(H):
        for j in range(W):
            if grid[i][j] != 0 and not visited[i][j]:
                color = grid[i][j]
                min_r, max_r = i, i
                min_c, max_c = j, j
                stack = [(i, j)]
                visited[i][j] = True
                while stack:
                    r, c = stack.pop()
                    min_r = min(min_r, r)
                    max_r = max(max_r, r)
                    min_c = min(min_c, c)
                    max_c = max(max_c, c)
                    for dr, dc in directions:
                        nr = r + dr
                        nc = c + dc
                        if 0 <= nr < H and 0 <= nc < W and not visited[nr][nc] and grid[nr][nc] == color:
                            visited[nr][nc] = True
                            stack.append((nr, nc))
                h_comp = max_r - min_r + 1
                w_comp = max_c - min_c + 1
                components.append((min_c, color, h_comp, w_comp))
    components.sort(key=lambda x: x[0])
    out = [[0] * W for _ in range(H)]
    if not components:
        return out
    # first component
    _, color, h, w = components[0]
    start_r = 0
    start_c = 0
    for dr in range(h):
        for dc in range(w):
            r = start_r + dr
            c = start_c + dc
            if 0 <= r < H and 0 <= c < W:
                out[r][c] = color
    attach_r = start_r + h - 1
    attach_c = start_c + w - 1
    # remaining components
    for _, color, h, w in components[1:]:
        start_r = attach_r
        start_c = attach_c
        for dr in range(h):
            for dc in range(w):
                r = start_r + dr
                c = start_c + dc
                if 0 <= r < H and 0 <= c < W:
                    out[r][c] = color
        attach_r = start_r + h - 1
        attach_c = start_c + w - 1
    return out