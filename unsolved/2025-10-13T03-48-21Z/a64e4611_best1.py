import numpy as np

def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    grid = np.array(grid_lst)
    rows, cols = grid.shape

    # Find the main color c
    colors = np.unique(grid)
    colors = colors[colors != 0]
    c = colors[0] if len(colors) > 0 else 0  # Assume one non-zero color

    # Find active columns
    active_cols = set()
    for col in range(cols):
        if np.any(grid[:, col] == c):
            active_cols.add(col)

    if not active_cols:
        return grid_lst

    sorted_cols = sorted(active_cols)

    # Find the largest gap
    max_gap = 0
    max_i = -1
    for i in range(len(sorted_cols) - 1):
        gap = sorted_cols[i + 1] - sorted_cols[i]
        if gap > max_gap:
            max_gap = gap
            max_i = i

    if max_i == -1:
        return grid_lst

    max_left = sorted_cols[max_i]
    min_right = sorted_cols[max_i + 1]

    start = max_left + 2
    end = min_right - 2

    if start > end:
        return grid_lst

    # Fill the stem in all rows
    for r in range(rows):
        for col in range(start, end + 1):
            if grid[r, col] == 0:
                grid[r, col] = 3

    # Define left and right column ranges
    left_columns = list(range(0, max_left + 1))
    right_columns = list(range(min_right, cols))

    # Function to find groups
    def find_groups(no_special_rows):
        groups = []
        i = 0
        while i < rows:
            if not no_special_rows[i]:
                i += 1
                continue
            j = i + 1
            while j < rows and no_special_rows[j]:
                j += 1
            if j - i >= 3:
                groups.append((i, j))
            i = j
        return groups

    # Right extension if applicable
    if c == 1 or c > 2:
        no_right_rows = [np.all(grid[r, right_columns] == 0) for r in range(rows)]
        groups = find_groups(no_right_rows)
        for g_start, g_end in groups:
            for r in range(g_start + 1, g_end - 1):
                for col in range(end + 1, cols):
                    if grid[r, col] == 0:
                        grid[r, col] = 3

    # Left extension if applicable
    if c == 2 or c > 2:
        no_left_rows = [np.all(grid[r, left_columns] == 0) for r in range(rows)]
        groups = find_groups(no_left_rows)
        for g_start, g_end in groups:
            for r in range(g_start + 1, g_end - 1):
                for col in range(0, start):
                    if grid[r, col] == 0:
                        grid[r, col] = 3

    return grid.tolist()