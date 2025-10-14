from collections import Counter, deque
from typing import List, Tuple

def transform(grid_lst: List[List[int]]) -> List[List[int]]:
    if not grid_lst or not grid_lst[0]:
        return []
    rows = len(grid_lst)
    cols = len(grid_lst[0])
    grid = [row[:] for row in grid_lst]

    # Find wall_color: most common non-zero
    flat_nonzero = [cell for row in grid for cell in row if cell != 0]
    if not flat_nonzero:
        return [row[:] for row in grid]
    wall_color = Counter(flat_nonzero).most_common(1)[0][0]

    # Label components of 0 cells
    label = [[-1 for _ in range(cols)] for _ in range(rows)]
    comp_cells: List[List[Tuple[int, int]]] = []
    comp_id = 0
    directions_connect = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 0 and label[r][c] == -1:
                cells = []
                stack = [(r, c)]
                label[r][c] = comp_id
                cells.append((r, c))
                while stack:
                    cr, cc = stack.pop()
                    for dr, dc in directions_connect:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 0 and label[nr][nc] == -1:
                            label[nr][nc] = comp_id
                            stack.append((nr, nc))
                            cells.append((nr, nc))
                comp_cells.append(cells)
                comp_id += 1

    num_comp = len(comp_cells)
    if num_comp == 0:
        return [row[:] for row in grid]

    # Build adjacency graph of components via wall cells
    adj: List[set] = [set() for _ in range(num_comp)]
    directions_wall = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 0:
                adj_comps = set()
                for dr, dc in directions_wall:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 0:
                        adj_comps.add(label[nr][nc])
                if len(adj_comps) >= 2:
                    clist = list(adj_comps)
                    for i in range(len(clist)):
                        for j in range(i + 1, len(clist)):
                            a, b = clist[i], clist[j]
                            adj[a].add(b)
                            adj[b].add(a)

    # Find starting component: top-left 0 cell
    start_r, start_c = 0, 0
    while start_r < rows and (start_c >= cols or grid[start_r][start_c] != 0):
        start_c += 1
        if start_c >= cols:
            start_c = 0
            start_r += 1
    if start_r >= rows:
        # No 0 cells
        return [row[:] for row in grid]
    start_comp = label[start_r][start_c]

    # Primary color for starting component
    primary = 4 if wall_color == 9 else 3
    opposite = 7 - primary

    # Color the components via BFS
    colors = [-1] * num_comp
    q = deque([start_comp])
    colors[start_comp] = primary
    visited = set([start_comp])

    while q:
        cur = q.popleft()
        for neigh in adj[cur]:
            if colors[neigh] == -1:
                colors[neigh] = opposite if colors[cur] == primary else primary
                q.append(neigh)
                visited.add(neigh)
            # Assume no conflict

    # For any unvisited components (disconnected), assign primary (or could choose differently, but assume all connected)
    for i in range(num_comp):
        if colors[i] == -1:
            colors[i] = primary

    # Apply colors
    out = [row[:] for row in grid]
    for cid in range(num_comp):
        col = colors[cid]
        for rr, cc in comp_cells[cid]:
            out[rr][cc] = col

    return out