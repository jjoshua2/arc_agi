import numpy as np
from collections import deque

def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    if not grid_lst:
        return []
    grid = np.array(grid_lst, dtype=int)
    rows, cols = grid.shape
    if rows == 0 or cols == 0:
        return grid.tolist()

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    visited = np.zeros((rows, cols), dtype=bool)

    # Mark all background 0s reachable from borders
    q = deque()
    for r in range(rows):
        if grid[r, 0] == 0 and not visited[r, 0]:
            q.append((r, 0))
            visited[r, 0] = True
        if grid[r, cols - 1] == 0 and not visited[r, cols - 1]:
            q.append((r, cols - 1))
            visited[r, cols - 1] = True
    for c in range(cols):
        if grid[0, c] == 0 and not visited[0, c]:
            q.append((0, c))
            visited[0, c] = True
        if grid[rows - 1, c] == 0 and not visited[rows - 1, c]:
            q.append((rows - 1, c))
            visited[rows - 1, c] = True

    while q:
        r, c = q.popleft()
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and not visited[nr, nc] and grid[nr, nc] == 0:
                visited[nr, nc] = True
                q.append((nr, nc))

    # Now find remaining 0 components (enclosed holes)
    for i in range(rows):
        for j in range(cols):
            if grid[i, j] == 0 and not visited[i, j]:
                # Flood fill to find component
                component = []
                comp_q = deque([(i, j)])
                visited[i, j] = True
                while comp_q:
                    cr, cc = comp_q.popleft()
                    component.append((cr, cc))
                    for dr, dc in directions:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < rows and 0 <= nc < cols and not visited[nr, nc] and grid[nr, nc] == 0:
                            visited[nr, nc] = True
                            comp_q.append((nr, nc))
                if not component:
                    continue
                # Compute bounding box
                min_r = min(pos[0] for pos in component)
                max_r = max(pos[0] for pos in component)
                min_c = min(pos[1] for pos in component)
                max_c = max(pos[1] for pos in component)
                h = max_r - min_r + 1
                w = max_c - min_c + 1
                # Check if solid rectangle
                if len(component) != h * w:
                    continue  # Not solid, skip (though examples are solid)
                # Fill with color
                color = 7 if (h % 2 == 1 and w % 2 == 1) else 2
                for rr in range(min_r, max_r + 1):
                    for cc in range(min_c, max_c + 1):
                        grid[rr, cc] = color

    return grid.tolist()