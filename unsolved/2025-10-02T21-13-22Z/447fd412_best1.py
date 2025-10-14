def transform(grid: list[list[int]]) -> list[list[int]]:
    if not grid or not grid[0]:
        return []
    rows = len(grid)
    cols = len(grid[0])
    output = [row[:] for row in grid]

    # Directions for 8-connected
    dirs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    # Find all components of 2's
    visited = [[False] * cols for _ in range(rows)]
    components = []
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == 2 and not visited[i][j]:
                component = []
                stack = [(i, j)]
                visited[i][j] = True
                while stack:
                    x, y = stack.pop()
                    component.append((x, y))
                    for dx, dy in dirs:
                        nx = x + dx
                        ny = y + dy
                        if 0 <= nx < rows and 0 <= ny < cols and not visited[nx][ny] and grid[nx][ny] == 2:
                            visited[nx][ny] = True
                            stack.append((nx, ny))
                components.append(component)

    if not components:
        return output

    # Main red component: largest one
    main_red_pos = max(components, key=len)
    if len(main_red_pos) <= 1:
        return output  # No main shape

    main_set = set(main_red_pos)

    # Collect seeds: all non-zero not in main
    seeds = []
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] != 0 and (i, j) not in main_set:
                seeds.append((i, j))

    if len(seeds) < 2:
        return output

    # Determine direction
    seed_rows = [s[0] for s in seeds]
    seed_cols = [s[1] for s in seeds]
    row_set = set(seed_rows)
    col_set = set(seed_cols)
    if len(row_set) == 1:
        direction = 'h'
        p_func = lambda s: s[1]
        fixed_perp = lambda pos: pos[0]  # row fixed to main's rows
    elif len(col_set) == 1:
        direction = 'v'
        p_func = lambda s: s[0]
        fixed_perp = lambda pos: pos[1]  # col fixed to main's cols
    else:
        return output  # Not aligned, assume not happen

    # Sort seeds by p
    zipped = sorted([(p_func(s), grid[s[0]][s[1]], s) for s in seeds])
    sorted_p = [z[0] for z in zipped]
    sorted_colors = [z[1] for z in zipped]
    sorted_seed_pos = [z[2] for z in zipped]
    m = len(zipped)

    # Find i_red: index of red seed (c==2)
    i_red_list = [i for i in range(m) if sorted_colors[i] == 2]
    if not i_red_list:
        return output  # No red seed
    i_red = i_red_list[0]  # assume one

    # Compute pc_red
    if direction == 'h':
        ps_main = [c for r, c in main_red_pos]
    else:
        ps_main = [r for r, c in main_red_pos]
    min_p_main = min(ps_main)
    max_p_main = max(ps_main)
    pc_red = min_p_main + (max_p_main - min_p_main) // 2

    # Compute pc for all
    pc = [0] * m
    pc[i_red] = pc_red

    # Forward
    for j in range(i_red + 1, m):
        diff = sorted_p[j] - sorted_p[j - 1]
        k = 1 if sorted_colors[j] == 2 else 2
        pc[j] = pc[j - 1] + diff + k

    # Backward
    for j in range(i_red - 1, -1, -1):
        diff = sorted_p[j + 1] - sorted_p[j]
        k = 1 if sorted_colors[j + 1] == 2 else 2
        pc[j] = pc[j + 1] - (diff + k)

    # Now place copies for non-red-seed
    for ii in range(m):
        if sorted_colors[ii] == 2 and ii == i_red:
            continue
        colr = sorted_colors[ii]
        delta = pc[ii] - pc_red
        if direction == 'h':
            for r_main, c_main in main_red_pos:
                new_c = c_main + delta
                if 0 <= new_c < cols:
                    output[r_main][new_c] = colr
        else:  # v
            for r_main, c_main in main_red_pos:
                new_r = r_main + delta
                if 0 <= new_r < rows:
                    output[new_r][c_main] = colr

    return output