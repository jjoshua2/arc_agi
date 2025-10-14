def transform(grid: list[list[int]]) -> list[list[int]]:
    if not grid or not grid[0]:
        return []
    rows = len(grid)
    cols = len(grid[0])
    output = [row[:] for row in grid]
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 2:
                for d in range(4):
                    dr, dc = directions[d]
                    sr = r + dr
                    sc = c + dc
                    if 0 <= sr < rows and 0 <= sc < cols and grid[sr][sc] == 4:
                        # Found starting cell sr, sc
                        # Vector from starting to 2
                        vec_r = r - sr
                        vec_c = c - sc
                        # Back direction from starting to 2
                        back_dr = vec_r
                        back_dc = vec_c
                        # Build path
                        path = [(sr, sc)]
                        current_r = sr
                        current_c = sc
                        visited = set([(sr, sc)])
                        added = True
                        while added:
                            added = False
                            for dd in range(4):
                                ddr, ddc = directions[dd]
                                nr = current_r + ddr
                                nc = current_c + ddc
                                if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 4 and (nr, nc) not in visited:
                                    # Check if not back
                                    if not (ddr == back_dr and ddc == back_dc):
                                        path.append((nr, nc))
                                        visited.add((nr, nc))
                                        current_r = nr
                                        current_c = nc
                                        back_dr = -ddr
                                        back_dc = -ddc
                                        added = True
                                        break
                        # Now path is built
                        # Check if straight horizontal or vertical
                        all_same_row = all(p[0] == path[0][0] for p in path)
                        all_same_col = all(p[1] == path[0][1] for p in path)
                        if not (all_same_row or all_same_col) or len(path) < 2:
                            continue
                        # Parallel or perpendicular
                        is_parallel = (all_same_row and vec_r == 0) or (all_same_col and vec_c == 0)
                        if is_parallel:
                            if all_same_row:  # horizontal parallel, add below
                                for k in range(1, len(path) + 1, 2):
                                    i = k - 1
                                    pr, pc = path[i]
                                    color = 1 if k % 4 == 1 else 3
                                    nr = pr + 1
                                    nc = pc
                                    if 0 <= nr < rows and 0 <= nc < cols and output[nr][nc] == 0:
                                        output[nr][nc] = color
                        else:
                            # Perpendicular case
                            if all_same_row:  # horizontal, sides up down
                                if vec_r > 0:  # 2 below, opposite up
                                    opposite_dr = -1
                                    same_dr = 1
                                elif vec_r < 0:  # 2 above, opposite down
                                    opposite_dr = 1
                                    same_dr = -1
                                else:
                                    continue
                                for i in range(len(path)):
                                    pr, pc = path[i]
                                    if i % 2 == 0:
                                        add_dr = opposite_dr
                                        color = 3
                                    else:
                                        add_dr = same_dr
                                        color = 1
                                    nr = pr + add_dr
                                    nc = pc
                                    if 0 <= nr < rows and 0 <= nc < cols and output[nr][nc] == 0:
                                        output[nr][nc] = color
                            elif all_same_col:  # vertical, sides left right
                                if vec_c > 0:  # 2 right, opposite left
                                    opposite_dc = -1
                                    same_dc = 1
                                elif vec_c < 0:  # 2 left, opposite right
                                    opposite_dc = 1
                                    same_dc = -1
                                else:
                                    continue
                                for i in range(len(path)):
                                    pr, pc = path[i]
                                    if i % 2 == 0:
                                        add_dc = opposite_dc
                                        color = 3
                                    else:
                                        add_dc = same_dc
                                        color = 1
                                    nr = pr
                                    nc = pc + add_dc
                                    if 0 <= nr < rows and 0 <= nc < cols and output[nr][nc] == 0:
                                        output[nr][nc] = color

    return output