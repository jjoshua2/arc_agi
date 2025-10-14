from collections import deque

def transform(grid_lst):
    if not grid_lst:
        return []
    h = len(grid_lst)
    if h == 0:
        return []
    w = len(grid_lst[0])
    grid = [row[:] for row in grid_lst]
    visited = [[False] * w for _ in range(h)]
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def flood_fill(start_r, start_c, target_color):
        queue = deque([(start_r, start_c)])
        vis = [[False] * w for _ in range(h)]
        vis[start_r][start_c] = True
        region = [(start_r, start_c)]
        while queue:
            r, c = queue.popleft()
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and not vis[nr][nc] and grid[nr][nc] == target_color:
                    vis[nr][nc] = True
                    queue.append((nr, nc))
                    region.append((nr, nc))
        return region

    for i in range(h):
        for j in range(w):
            if grid[i][j] == 2 and not visited[i][j]:
                # Find red component using DFS
                comp = []
                stack = [(i, j)]
                visited[i][j] = True
                while stack:
                    r, c = stack.pop()
                    comp.append((r, c))
                    for dr, dc in directions:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < h and 0 <= nc < w and grid[nr][nc] == 2 and not visited[nr][nc]:
                            visited[nr][nc] = True
                            stack.append((nr, nc))
                if not comp:
                    continue
                min_r = min(r for r, c in comp)
                max_r = max(r for r, c in comp)
                min_c = min(c for r, c in comp)
                max_c = max(c for r, c in comp)
                hh = max_r - min_r + 1
                ww = max_c - min_c + 1
                if len(comp) != hh * ww:
                    continue  # Assume solid rectangle
                # Find adjacent non-2 cells
                adj_colors = set()
                adj_positions = set()  # Use set to avoid duplicates
                for r, c in comp:
                    for dr, dc in directions:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < h and 0 <= nc < w and grid[nr][nc] != 2:
                            adj_colors.add(grid[nr][nc])
                            adj_positions.add((nr, nc))
                if len(adj_colors) != 1 or list(adj_colors)[0] not in (1, 8):
                    continue  # Assume valid
                B = list(adj_colors)[0]
                # Pick one adjacent position
                start_r, start_c = next(iter(adj_positions))
                # Flood fill the region of B
                region = flood_fill(start_r, start_c, B)
                reg_min_c = min(c for _, c in region)
                reg_max_c = max(c for _, c in region)
                # Compute new column range
                if B == 1:
                    new_min_c = reg_max_c - ww + 1
                    new_max_c = reg_max_c
                else:
                    new_min_c = reg_min_c
                    new_max_c = reg_min_c + ww - 1
                # Set original positions to B
                for r, c in comp:
                    grid[r][c] = B
                # Set new positions to 2
                for r in range(min_r, max_r + 1):
                    for cc in range(new_min_c, new_max_c + 1):
                        if 0 <= cc < w:
                            grid[r][cc] = 2
    return grid