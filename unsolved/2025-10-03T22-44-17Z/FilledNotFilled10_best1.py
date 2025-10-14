def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    if not grid_lst or not grid_lst[0]:
        return grid_lst

    # Work on a copy to avoid mutating the input
    grid = [row[:] for row in grid_lst]
    H = len(grid)
    W = len(grid[0])

    # Find connected components (non-zero, same color)
    from collections import deque

    directions = [(0,1),(1,0),(0,-1),(-1,0)]
    visited = [[False]*W for _ in range(H)]
    components = []

    for r in range(H):
        for c in range(W):
            if grid[r][c] != 0 and not visited[r][c]:
                color = grid[r][c]
                q = deque([(r,c)])
                visited[r][c] = True
                cells = []
                while q:
                    cr, cc = q.popleft()
                    cells.append((cr, cc))
                    for dr, dc in directions:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < H and 0 <= nc < W and not visited[nr][nc] and grid[nr][nc] == color:
                            visited[nr][nc] = True
                            q.append((nr, nc))
                components.append({'color': color, 'cells': cells})

    if not components:
        return grid

    # Central shape is the largest component
    central = max(components, key=lambda x: len(x['cells']))
    central_color = central['color']
    min_r = min(r for r, _ in central['cells'])
    max_r = max(r for r, _ in central['cells'])
    min_c = min(c for _, c in central['cells'])
    max_c = max(c for _, c in central['cells'])

    # Prepare output
    output = [row[:] for row in grid]

    # Map stray cells to edge cells
    for r in range(H):
        for c in range(W):
            val = grid[r][c]
            if val == 0 or val == central_color:
                continue

            # Above/below alignment (same column)
            if min_c <= c <= max_c:
                if r < min_r:
                    tr, tc = min_r, c
                    if 0 <= tr < H and 0 <= tc < W and output[tr][tc] == 0:
                        output[tr][tc] = val
                elif r > max_r:
                    tr, tc = max_r, c
                    if 0 <= tr < H and 0 <= tc < W and output[tr][tc] == 0:
                        output[tr][tc] = val

            # Left/right alignment (same row)
            if min_r <= r <= max_r:
                if c < min_c:
                    tr, tc = r, min_c
                    if 0 <= tr < H and 0 <= tc < W and output[tr][tc] == 0:
                        output[tr][tc] = val
                elif c > max_c:
                    tr, tc = r, max_c
                    if 0 <= tr < H and 0 <= tc < W and output[tr][tc] == 0:
                        output[tr][tc] = val

    return output