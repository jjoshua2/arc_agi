from collections import deque, defaultdict
import copy

def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    if not grid_lst or not grid_lst[0]:
        return []
    grid = [row[:] for row in grid_lst]
    rows = len(grid)
    cols = len(grid[0])

    # Function to get components
    def get_nonzero_components(g):
        visited = [[False] * cols for _ in range(rows)]
        components = []
        for i in range(rows):
            for j in range(cols):
                if g[i][j] != 0 and not visited[i][j]:
                    color = g[i][j]
                    comp = []
                    queue = deque([(i, j)])
                    visited[i][j] = True
                    while queue:
                        x, y = queue.popleft()
                        comp.append((x, y))
                        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < rows and 0 <= ny < cols and not visited[nx][ny] and g[nx][ny] == color:
                                visited[nx][ny] = True
                                queue.append((nx, ny))
                    components.append((color, comp))
        return components

    # Get original components
    orig_components = get_nonzero_components(grid)

    # Small if size <= 2
    small_positions = set()
    for _, comp in orig_components:
        if len(comp) <= 2:
            for p in comp:
                small_positions.add(p)

    # Temp grid: set small to 0
    temp = [row[:] for row in grid]
    for r, c in small_positions:
        temp[r][c] = 0

    # Get 0 components in temp
    zero_components = []
    visited = [[False] * cols for _ in range(rows)]
    for i in range(rows):
        for j in range(cols):
            if temp[i][j] == 0 and not visited[i][j]:
                comp = []
                queue = deque([(i, j)])
                visited[i][j] = True
                while queue:
                    x, y = queue.popleft()
                    comp.append((x, y))
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < rows and 0 <= ny < cols and temp[nx][ny] == 0 and not visited[nx][ny]:
                            visited[nx][ny] = True
                            queue.append((nx, ny))
                zero_components.append(comp)

    # Determine if each zero comp is exterior
    is_exterior = [False] * len(zero_components)
    for idx, comp in enumerate(zero_components):
        if len(comp) <= 1:
            is_exterior[idx] = True
            continue
        touches = False
        for r, c in comp:
            if r == 0 or r == rows - 1 or c == 0 or c == cols - 1:
                touches = True
                break
        is_exterior[idx] = touches

    # comp_id for each position
    comp_id = [[-1] * cols for _ in range(rows)]
    for idx, comp in enumerate(zero_components):
        for r, c in comp:
            comp_id[r][c] = idx

    # Output grid
    output = [row[:] for row in grid]

    # Process small positions
    for r, c in small_positions:
        idx = comp_id[r][c]
        if idx == -1:
            continue
        if not is_exterior[idx]:
            output[r][c] = 0
            continue
        # exterior, count original neighbors
        counts = defaultdict(int)
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                colr = grid[nr][nc]
                counts[colr] += 1
        # Find max_cnt and largest d with that cnt
        max_cnt = 0
        best_d = 0
        for d in sorted(counts.keys(), reverse=True):
            cnt = counts[d]
            if cnt > max_cnt:
                max_cnt = cnt
                best_d = d
            elif cnt == max_cnt and d > best_d:
                best_d = d
        if max_cnt >= 2 and best_d != 0:
            output[r][c] = best_d
        else:
            output[r][c] = 0

    return output