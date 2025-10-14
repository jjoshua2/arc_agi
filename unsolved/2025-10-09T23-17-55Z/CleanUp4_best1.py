from collections import deque
from typing import List, Tuple, Dict, Set

def transform(grid: List[List[int]]) -> List[List[int]]:
    if not grid or not grid[0]:
        return []
    rows = len(grid)
    cols = len(grid[0])
    out = [row[:] for row in grid]
    dirs: List[Tuple[int, int]] = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    def is_border(r: int, c: int) -> bool:
        return r == 0 or r == rows - 1 or c == 0 or c == cols - 1

    def get_components(g: List[List[int]]) -> List[Tuple[int, List[Tuple[int, int]]]]:
        visited = [[False] * cols for _ in range(rows)]
        components = []
        for i in range(rows):
            for j in range(cols):
                if g[i][j] > 0 and not visited[i][j]:
                    color = g[i][j]
                    component = []
                    stack = [(i, j)]
                    visited[i][j] = True
                    while stack:
                        x, y = stack.pop()
                        component.append((x, y))
                        for dx, dy in dirs:
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < rows and 0 <= ny < cols and not visited[nx][ny] and g[nx][ny] == color:
                                visited[nx][ny] = True
                                stack.append((nx, ny))
                    components.append((color, component))
        return components

    # Get persistent colors: colors with at least one component of size >= 3
    comps = get_components(out)
    persistent: Set[int] = set()
    for colr, comp in comps:
        if len(comp) >= 3:
            persistent.add(colr)

    # Find size 1 components of non-persistent colors
    merges: Dict[Tuple[int, int], int] = {}
    comps = get_components(out)
    for colr, comp in comps:
        if colr not in persistent and len(comp) == 1:
            r, c = comp[0]
            enclosed = False
            for target_c in persistent:
                # BFS to check if can reach border through non-target_c cells
                visited = [[False] * cols for _ in range(rows)]
                q = deque()
                q.append((r, c))
                visited[r][c] = True
                reached_border = is_border(r, c)
                while q and not reached_border:
                    x, y = q.popleft()
                    for dx, dy in dirs:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < rows and 0 <= ny < cols and not visited[nx][ny] and out[nx][ny] != target_c:
                            visited[nx][ny] = True
                            q.append((nx, ny))
                            if is_border(nx, ny):
                                reached_border = True
                                break
                    if reached_border:
                        break
                if not reached_border:
                    enclosed = True
                    break
            if enclosed:
                continue
            # Count adjacent persistent C cells
            adj: Dict[int, int] = {pc: 0 for pc in persistent}
            for dx, dy in dirs:
                nr, nc = r + dx, c + dy
                if 0 <= nr < rows and 0 <= nc < cols and out[nr][nc] in persistent:
                    adj[out[nr][nc]] += 1
            # Find C with max adj >=2
            max_adj = 0
            chosen_c = None
            for pc, cnt in adj.items():
                if cnt > max_adj:
                    max_adj = cnt
                    chosen_c = pc
            if max_adj >= 2:
                merges[(r, c)] = chosen_c

    # Apply merges
    for (r, c), tc in merges.items():
        out[r][c] = tc

    # Now recompute components and clear small ones <3
    comps = get_components(out)
    to_clear: Set[Tuple[int, int]] = set()
    for colr, comp in comps:
        if len(comp) < 3:
            for pos in comp:
                to_clear.add(pos)
    for (r, c) in to_clear:
        out[r][c] = 0

    return out