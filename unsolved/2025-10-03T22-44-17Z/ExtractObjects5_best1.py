def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    if not grid_lst or not grid_lst[0]:
        return []
    m = len(grid_lst)
    n = len(grid_lst[0])
    grid = grid_lst  # No need to copy

    pluses = []
    for r in range(m - 3):
        for c in range(n - 3):
            # Check if this is a plus
            if grid[r][c] == 0:
                continue
            color = grid[r][c]
            is_plus = True
            # Check all border positions == color
            for i in range(4):
                for j in range(4):
                    if 1 <= i <= 2 and 1 <= j <= 2:
                        # Inner must be 0
                        if grid[r + i][c + j] != 0:
                            is_plus = False
                            break
                    else:
                        # Border must be color
                        if grid[r + i][c + j] != color:
                            is_plus = False
                            break
                if not is_plus:
                    break
            if is_plus:
                row_center = r + 1.5
                col_center = c + 1.5
                pluses.append({
                    'color': color,
                    'row_center': row_center,
                    'col_center': col_center
                })

    num = len(pluses)
    if num == 0:
        return []

    row_centers = [p['row_center'] for p in pluses]
    col_centers = [p['col_center'] for p in pluses]
    row_spread = max(row_centers) - min(row_centers)
    col_spread = max(col_centers) - min(col_centers)

    def make_plus(color):
        plus = [[0] * 4 for _ in range(4)]
        for i in range(4):
            for j in range(4):
                if not (1 <= i <= 2 and 1 <= j <= 2):
                    plus[i][j] = color
        return plus

    if col_spread > row_spread:
        # Horizontal arrangement
        sorted_pluses = sorted(pluses, key=lambda p: p['col_center'])
        out_height = 4
        out_width = 4 * num
        output = [[0] * out_width for _ in range(out_height)]
        for idx, p in enumerate(sorted_pluses):
            plus = make_plus(p['color'])
            start_col = idx * 4
            for i in range(4):
                for j in range(4):
                    output[i][start_col + j] = plus[i][j]
    else:
        # Vertical arrangement
        sorted_pluses = sorted(pluses, key=lambda p: p['row_center'])
        out_height = 4 * num
        out_width = 4
        output = [[0] * out_width for _ in range(out_height)]
        for idx, p in enumerate(sorted_pluses):
            plus = make_plus(p['color'])
            start_row = idx * 4
            for i in range(4):
                for j in range(4):
                    output[start_row + i][j] = plus[i][j]

    return output