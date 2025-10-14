from collections import deque

def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    if not grid_lst:
        return []
    h = len(grid_lst)
    if h == 0:
        return []
    w = len(grid_lst[0])
    output = [row[:] for row in grid_lst]
    visited = [[False] * w for _ in range(h)]
    q = deque()
    # Add all border 0s to queue
    for r in range(h):
        if grid_lst[r][0] == 0 and not visited[r][0]:
            q.append((r, 0))
            visited[r][0] = True
        if grid_lst[r][w - 1] == 0 and not visited[r][w - 1]:
            q.append((r, w - 1))
            visited[r][w - 1] = True
    for c in range(w):
        if grid_lst[0][c] == 0 and not visited[0][c]:
            q.append((0, c))
            visited[0][c] = True
        if grid_lst[h - 1][c] == 0 and not visited[h - 1][c]:
            q.append((h - 1, c))
            visited[h - 1][c] = True
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    while q:
        r, c = q.popleft()
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not visited[nr][nc] and grid_lst[nr][nc] == 0:
                visited[nr][nc] = True
                q.append((nr, nc))
    # Fill unvisited 0s (enclosed) with 2
    for r in range(h):
        for c in range(w):
            if grid_lst[r][c] == 0 and not visited[r][c]:
                output[r][c] = 2
    return output