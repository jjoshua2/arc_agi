from collections import deque

def transform(grid: list[list[int]]) -> list[list[int]]:
    if not grid or not grid[0]:
        return []
    h = len(grid)
    w = len(grid[0])
    out = [row[:] for row in grid]
    visited = [[False for _ in range(w)] for _ in range(h)]
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    for r in range(h):
        for c in range(w):
            if grid[r][c] != 5 and not visited[r][c]:
                component = []
                has_two = False
                queue = deque([(r, c)])
                visited[r][c] = True
                while queue:
                    cr, cc = queue.popleft()
                    component.append((cr, cc))
                    if grid[cr][cc] == 2:
                        has_two = True
                    for dr, dc in directions:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < h and 0 <= nc < w and not visited[nr][nc] and grid[nr][nc] != 5:
                            visited[nr][nc] = True
                            queue.append((nr, nc))
                if has_two:
                    for pr, pc in component:
                        if out[pr][pc] == 0:
                            out[pr][pc] = 2
    return out