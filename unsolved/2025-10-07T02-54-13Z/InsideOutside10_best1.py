from collections import deque

def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    if not grid_lst or not grid_lst[0]:
        return grid_lst
    grid = [row[:] for row in grid_lst]
    rows = len(grid)
    cols = len(grid[0])
    # Find non-zero colors and P (touching perimeter)
    color_set = set()
    perimeter_colors = set()
    for i in range(rows):
        for j in range(cols):
            c = grid[i][j]
            if c != 0:
                color_set.add(c)
            if i == 0 or i == rows - 1 or j == 0 or j == cols - 1:
                if c != 0:
                    perimeter_colors.add(c)
    if len(color_set) != 2 or len(perimeter_colors) != 1:
        return grid  # Assume exactly two colors, one touches perimeter
    p = list(perimeter_colors)[0]
    f = (color_set - perimeter_colors).pop()
    # Step 1: Handle sandwiched P to F
    changed = True
    while changed:
        changed = False
        for i in range(rows):
            for j in range(cols):
                if grid[i][j] == p:
                    # Vertical sandwiched
                    if i > 0 and i < rows - 1 and grid[i - 1][j] == f and grid[i + 1][j] == f:
                        grid[i][j] = f
                        changed = True
                    # Horizontal sandwiched
                    if j > 0 and j < cols - 1 and grid[i][j - 1] == f and grid[i][j + 1] == f:
                        grid[i][j] = f
                        changed = True
    # Step 2: Border trim BFS
    visited = [[False] * cols for _ in range(rows)]
    q = deque()
    # Add perimeter P cells as sources
    for i in range(rows):
        for j in [0, cols - 1]:
            if grid[i][j] == p and not visited[i][j]:
                visited[i][j] = True
                q.append((i, j, 0))
    for j in range(cols):
        for i in [0, rows - 1]:
            if grid[i][j] == p and not visited[i][j]:
                visited[i][j] = True
                q.append((i, j, 0))
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    while q:
        r, c, dist = q.popleft()
        if grid[r][c] != p:
            continue
        # Check 8-adj to F
        adj_f = False
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                if dr == 0 and dc == 0:
                    continue
                nr = r + dr
                nc = c + dc
                if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == f:
                    adj_f = True
                    break
            if adj_f:
                break
        enqueue = False
        if dist == 0:
            if adj_f:
                grid[r][c] = f
            else:
                grid[r][c] = 0
                enqueue = True
        elif dist == 1:
            grid[r][c] = 0
            enqueue = True
        else:
            if adj_f:
                grid[r][c] = f
            else:
                grid[r][c] = 0
                enqueue = True
        if enqueue:
            for dr, dc in dirs:
                nr = r + dr
                nc = c + dc
                if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == p and not visited[nr][nc]:
                    visited[nr][nc] = True
                    q.append((nr, nc, dist + 1))
    # Step 3: Inner trim
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == p:
                for dr, dc in dirs:
                    ni = i + dr
                    nj = j + dc
                    if 0 <= ni < rows and 0 <= nj < cols and grid[ni][nj] == 0:
                        nni = ni + dr
                        nnj = nj + dc
                        if 0 <= nni < rows and 0 <= nnj < cols and grid[nni][nnj] == p:
                            # Check if beyond P adj to F (8-dir)
                            b_adj_f = False
                            for ddr in range(-1, 2):
                                for ddc in range(-1, 2):
                                    if ddr == 0 and ddc == 0:
                                        continue
                                    bni = nni + ddr
                                    bnj = nnj + ddc
                                    if 0 <= bni < rows and 0 <= bnj < cols and grid[bni][bnj] == f:
                                        b_adj_f = True
                                        break
                                if b_adj_f:
                                    break
                            if not b_adj_f:
                                grid[i][j] = 0
                                break  # Trim this cell
    return grid