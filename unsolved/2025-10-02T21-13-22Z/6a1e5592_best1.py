def find_components(grid):
    if not grid or not grid[0]:
        return []
    rows = len(grid)
    cols = len(grid[0])
    visited = set()
    components = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 5 and (r, c) not in visited:
                component = []
                stack = [(r, c)]
                visited.add((r, c))
                while stack:
                    cr, cc = stack.pop()
                    component.append((cr, cc))
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 5 and (nr, nc) not in visited:
                            visited.add((nr, nc))
                            stack.append((nr, nc))
                components.append(component)
    return components

def transform(grid_lst):
    if not grid_lst or not grid_lst[0]:
        return []
    rows = len(grid_lst)
    cols = len(grid_lst[0])
    output = [row[:] for row in grid_lst]
    components = find_components(grid_lst)
    for comp in components:
        poss = comp
        minr = min(p[0] for p in poss)
        maxr = max(p[0] for p in poss)
        minc = min(p[1] for p in poss)
        maxc = max(p[1] for p in poss)
        height = maxr - minr + 1
        width = maxc - minc + 1
        half_h = height // 2
        left_w = width // 2
        above_r = minr - 1
        below_r = maxr + 1
        left_above = 0
        right_above = 0
        left_below = 0
        right_below = 0
        if above_r >= 0:
            lcol = minc - 1
            if 0 <= lcol < cols:
                left_above = grid_lst[above_r][lcol]
            rcol = maxc + 1
            if 0 <= rcol < cols:
                right_above = grid_lst[above_r][rcol]
        if below_r < rows:
            lcol = minc - 1
            if 0 <= lcol < cols:
                left_below = grid_lst[below_r][lcol]
            rcol = maxc + 1
            if 0 <= rcol < cols:
                right_below = grid_lst[below_r][rcol]
        # Fill upper half
        for i in range(minr, minr + half_h):
            for j in range(minc, minc + left_w):
                output[i][j] = left_above
            for j in range(minc + left_w, maxc + 1):
                output[i][j] = right_above
        # Fill lower half
        for i in range(minr + half_h, maxr + 1):
            for j in range(minc, minc + left_w):
                output[i][j] = left_below
            for j in range(minc + left_w, maxc + 1):
                output[i][j] = right_below
        # Set border colors to 0
        if above_r >= 0:
            lcol = minc - 1
            if 0 <= lcol < cols and grid_lst[above_r][lcol] != 0:
                output[above_r][lcol] = 0
            rcol = maxc + 1
            if 0 <= rcol < cols and grid_lst[above_r][rcol] != 0:
                output[above_r][rcol] = 0
        if below_r < rows:
            lcol = minc - 1
            if 0 <= lcol < cols and grid_lst[below_r][lcol] != 0:
                output[below_r][lcol] = 0
            rcol = maxc + 1
            if 0 <= rcol < cols and grid_lst[below_r][rcol] != 0:
                output[below_r][rcol] = 0
    return output