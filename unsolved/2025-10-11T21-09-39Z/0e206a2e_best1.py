from collections import Counter, defaultdict
import copy

def get_majority(grid):
    if not grid or not grid[0]:
        return None
    flat = [grid[r][c] for r in range(len(grid)) for c in range(len(grid[0])) if grid[r][c] != 0]
    if not flat:
        return None
    count = Counter(flat)
    return count.most_common(1)[0][0]

def get_components(grid):
    rows = len(grid)
    cols = len(grid[0])
    visited = [[False] * cols for _ in range(rows)]
    components = []
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] != 0 and not visited[i][j]:
                component = []
                stack = [(i, j)]
                visited[i][j] = True
                while stack:
                    x, y = stack.pop()
                    component.append((x, y))
                    for dx, dy in directions:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < rows and 0 <= ny < cols and grid[nx][ny] != 0 and not visited[nx][ny]:
                            visited[nx][ny] = True
                            stack.append((nx, ny))
                components.append(component)
    return components

def transform(grid_lst):
    if not grid_lst or not grid_lst[0]:
        return []
    grid = copy.deepcopy(grid_lst)
    maj = get_majority(grid)
    if maj is None:
        return grid
    comps = get_components(grid)
    terminal_positions = set()
    for comp in comps:
        has_maj = any(grid[r][c] == maj for r, c in comp)
        if has_maj:
            for r, c in comp:
                grid[r][c] = 0
        else:
            for r, c in comp:
                terminal_positions.add((r, c))
    # Now terminals
    terms = list(terminal_positions)
    n = len(terms)
    if n == 0:
        return grid
    # Build graph for clusters
    graph = defaultdict(list)
    for i in range(n):
        r1, c1 = terms[i]
        for j in range(i + 1, n):
            r2, c2 = terms[j]
            dist = abs(r1 - r2) + abs(c1 - c2)
            if 0 < dist <= 5:
                graph[i].append(j)
                graph[j].append(i)
    # Find clusters
    visited = [False] * n
    clusters = []
    for i in range(n):
        if not visited[i]:
            cluster = []
            stack = [i]
            visited[i] = True
            while stack:
                u = stack.pop()
                cluster.append(u)
                for v in graph[u]:
                    if not visited[v]:
                        visited[v] = True
                        stack.append(v)
            if len(cluster) == 3:
                clust_terms = [terms[idx] for idx in cluster]
                clust_terms.sort(key=lambda p: (p[0], p[1]))
                clusters.append(clust_terms)
    # Process each cluster
    rows_len = len(grid)
    cols_len = len(grid[0])
    for clust in clusters:
        p1 = clust[0]
        p2 = clust[1]
        p3 = clust[2]
        # Check for horizontal pair
        if p2[0] == p3[0]:
            # horizontal base at p2 p3 row, p1 offset upper
            base_row = p2[0]
            pair_cols = [p2[1], p3[1]]
            L = min(pair_cols)
            R = max(pair_cols)
            offset_r, offset_c = p1
            delta = base_row - offset_r
            # horizontal fill
            for cc in range(L + 1, R):
                if 0 <= cc < cols_len:
                    grid[base_row][cc] = maj
            # vertical fill
            minr = offset_r
            maxr = base_row
            for rr in range(minr + 1, maxr):
                if 0 <= rr < rows_len:
                    grid[rr][offset_c] = maj
            if delta >= 2:
                pos1_r, pos1_c = offset_r, L + 1
                if 0 <= pos1_c < cols_len and (pos1_r, pos1_c) not in terminal_positions:
                    grid[pos1_r][pos1_c] = maj
                pos2_r, pos2_c = offset_r, R - 1
                if 0 <= pos2_c < cols_len and (pos2_r, pos2_c) not in terminal_positions:
                    grid[pos2_r][pos2_c] = maj
            else:
                extended_r = base_row + 1
                # full vert L
                for rr in range(offset_r, extended_r + 1):
                    if 0 <= rr < rows_len and (rr, L) not in terminal_positions:
                        grid[rr][L] = maj
                # full vert offset_c
                for rr in range(offset_r, extended_r + 1):
                    if 0 <= rr < rows_len and (rr, offset_c) not in terminal_positions:
                        grid[rr][offset_c] = maj
        elif p1[0] == p2[0]:
            # horizontal base at p1 p2 row, p3 offset lower
            base_row = p1[0]
            pair_cols = [p1[1], p2[1]]
            L = min(pair_cols)
            R = max(pair_cols)
            offset_r, offset_c = p3
            delta = offset_r - base_row
            # horizontal fill
            for cc in range(L + 1, R):
                if 0 <= cc < cols_len:
                    grid[base_row][cc] = maj
            # vertical fill
            minr = base_row
            maxr = offset_r
            for rr in range(minr + 1, maxr):
                if 0 <= rr < rows_len:
                    grid[rr][offset_c] = maj
            if delta >= 2:
                for rr in range(base_row + 1, offset_r):
                    if 0 <= rr < rows_len:
                        pos_l = (rr, L)
                        if pos_l not in terminal_positions:
                            grid[rr][L] = maj
                        pos_r = (rr, R)
                        if pos_r not in terminal_positions:
                            grid[rr][R] = maj
                # extra horizontal
                mid = (L + R) / 2
                if offset_c >= mid:
                    extra_c = R + 1
                    if 0 <= extra_c < cols_len and (base_row, extra_c) not in terminal_positions:
                        grid[base_row][extra_c] = maj
                else:
                    extra_c = L - 1
                    if 0 <= extra_c < cols_len and (base_row, extra_c) not in terminal_positions:
                        grid[base_row][extra_c] = maj
            else:
                extended_r = base_row - 1
                # full vert L
                for rr in range(extended_r, offset_r + 1):
                    if 0 <= rr < rows_len and (rr, L) not in terminal_positions:
                        grid[rr][L] = maj
                # full vert offset_c
                for rr in range(extended_r, offset_r + 1):
                    if 0 <= rr < rows_len and (rr, offset_c) not in terminal_positions:
                        grid[rr][offset_c] = maj
        else:
            # vertical pair
            # find which two share col
            if p1[1] == p2[1]:
                main_col = p1[1]
                top_r = min(p1[0], p2[0])
                bot_r = max(p1[0], p2[0])
                offset_r = p3[0]
                offset_c = p3[1]
            elif p1[1] == p3[1]:
                main_col = p1[1]
                top_r = min(p1[0], p3[0])
                bot_r = max(p1[0], p3[0])
                offset_r = p2[0]
                offset_c = p2[1]
            elif p2[1] == p3[1]:
                main_col = p2[1]
                top_r = min(p2[0], p3[0])
                bot_r = max(p2[0], p3[0])
                offset_r = p1[0]
                offset_c = p1[1]
            else:
                continue  # no pair
            # now
            if top_r < offset_r < bot_r:
                # full mode
                span_top = top_r
                span_bot = bot_r
                # vertical main
                for rr in range(span_top, span_bot + 1):
                    if 0 <= rr < rows_len and (rr, main_col) not in terminal_positions:
                        grid[rr][main_col] = maj
                # vertical offset
                for rr in range(span_top, span_bot + 1):
                    if 0 <= rr < rows_len and (rr, offset_c) not in terminal_positions:
                        grid[rr][offset_c] = maj
                # horizontal
                left_c = min(main_col, offset_c) - 1
                right_c = max(main_col, offset_c)
                for cc in range(left_c, right_c + 1):
                    if 0 <= cc < cols_len and (offset_r, cc) not in terminal_positions:
                        grid[offset_r][cc] = maj
            else:
                if offset_r < top_r:
                    # upper offset
                    span_top = offset_r
                    span_bot = bot_r
                    # vertical main
                    for rr in range(span_top, span_bot + 1):
                        if 0 <= rr < rows_len and (rr, main_col) not in terminal_positions:
                            grid[rr][main_col] = maj
                    # horizontal at offset_r
                    left_c = min(main_col, offset_c)
                    right_c = max(main_col, offset_c)
                    for cc in range(left_c, right_c + 1):
                        if 0 <= cc < cols_len and (offset_r, cc) not in terminal_positions:
                            grid[offset_r][cc] = maj
                    # extra
                    if offset_c > main_col:
                        extra_c = offset_c + 1
                        if 0 <= extra_c < cols_len and (offset_r, extra_c) not in terminal_positions:
                            grid[offset_r][extra_c] = maj
                    else:
                        extra_c = offset_c - 1
                        if 0 <= extra_c < cols_len and (offset_r, extra_c) not in terminal_positions:
                            grid[offset_r][extra_c] = maj
                else:
                    # lower offset
                    span_top = top_r
                    span_bot = offset_r
                    # vertical main
                    for rr in range(span_top, span_bot + 1):
                        if 0 <= rr < rows_len and (rr, main_col) not in terminal_positions:
                            grid[rr][main_col] = maj
                    # horizontal at offset_r
                    left_c = min(main_col, offset_c)
                    right_c = max(main_col, offset_c)
                    for cc in range(left_c, right_c + 1):
                        if 0 <= cc < cols_len and (offset_r, cc) not in terminal_positions:
                            grid[offset_r][cc] = maj
                    # extra
                    if offset_c > main_col:
                        extra_c = offset_c + 1
                        if 0 <= extra_c < cols_len and (offset_r, extra_c) not in terminal_positions:
                            grid[offset_r][extra_c] = maj
                    else:
                        extra_c = offset_c - 1
                        if 0 <= extra_c < cols_len and (offset_r, extra_c) not in terminal_positions:
                            grid[offset_r][extra_c] = maj
    return grid