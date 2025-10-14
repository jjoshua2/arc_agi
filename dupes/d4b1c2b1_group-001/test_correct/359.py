def get_intervals(size: int, walls: list[int]) -> list[tuple[int, int]]:
    sorted_walls = sorted(set(walls))
    intervals = []
    prev = -1
    for w in sorted_walls + [size]:
        start = prev + 1
        end = w - 1
        if start <= end:
            intervals.append((start, end))
        prev = w
    return intervals

def transform(grid: list[list[int]]) -> list[list[int]]:
    if not grid or not grid[0]:
        return []
    rows = len(grid)
    cols = len(grid[0])
    output = [row[:] for row in grid]
    
    # Find wall_rows: full rows of 4
    wall_rows = [i for i in range(rows) if all(grid[i][j] == 4 for j in range(cols))]
    
    # Find wall_cols: full columns of 4
    wall_cols = [j for j in range(cols) if all(grid[i][j] == 4 for i in range(rows))]
    
    panels = get_intervals(rows, wall_rows)
    strips = get_intervals(cols, wall_cols)
    
    if rows > cols:  # Vertical mode: propagate left-right within each panel
        if not wall_cols:
            return output
        wall_c = wall_cols[0]  # Assume single wall
        left_start, left_end = 0, wall_c - 1
        right_start, right_end = wall_c + 1, cols - 1
        for p_start, p_end in panels:
            left_colors = set()
            for i in range(p_start, p_end + 1):
                for j in range(left_start, left_end + 1):
                    c = grid[i][j]
                    if c not in {0, 1, 4}:
                        left_colors.add(c)
            right_colors = set()
            for i in range(p_start, p_end + 1):
                for j in range(right_start, right_end + 1):
                    c = grid[i][j]
                    if c not in {0, 1, 4}:
                        right_colors.add(c)
            if len(left_colors) == 1 and len(right_colors) == 0:
                c = next(iter(left_colors))
                for i in range(p_start, p_end + 1):
                    for j in range(right_start, right_end + 1):
                        if output[i][j] == 1:
                            output[i][j] = c
            elif len(left_colors) == 0 and len(right_colors) == 1:
                c = next(iter(right_colors))
                for i in range(p_start, p_end + 1):
                    for j in range(left_start, left_end + 1):
                        if output[i][j] == 1:
                            output[i][j] = c
    else:  # Horizontal mode: propagate up-down within each strip (assume 2 panels)
        if len(panels) != 2:
            return output
        upper_start, upper_end = panels[0]
        lower_start, lower_end = panels[1]
        for s_start, s_end in strips:
            upper_colors = set()
            for i in range(upper_start, upper_end + 1):
                for j in range(s_start, s_end + 1):
                    c = grid[i][j]
                    if c not in {0, 1, 4}:
                        upper_colors.add(c)
            lower_colors = set()
            for i in range(lower_start, lower_end + 1):
                for j in range(s_start, s_end + 1):
                    c = grid[i][j]
                    if c not in {0, 1, 4}:
                        lower_colors.add(c)
            if len(upper_colors) == 1 and len(lower_colors) == 0:
                c = next(iter(upper_colors))
                for i in range(lower_start, lower_end + 1):
                    for j in range(s_start, s_end + 1):
                        if output[i][j] == 1:
                            output[i][j] = c
            elif len(lower_colors) == 1 and len(upper_colors) == 0:
                c = next(iter(lower_colors))
                for i in range(upper_start, upper_end + 1):
                    for j in range(s_start, s_end + 1):
                        if output[i][j] == 1:
                            output[i][j] = c
    
    return output