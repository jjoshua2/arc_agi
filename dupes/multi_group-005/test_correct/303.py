def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    if len(grid_lst) != 2 or len(grid_lst[0]) != 12:
        raise ValueError("Input must be 2x12 grid")
    grid = grid_lst
    C_cols = [j for j in range(12) if grid[0][j] == 0 and grid[1][j] == 2]
    # Build groups where consecutive C's have diff <=2
    groups = []
    if C_cols:
        current_group = [C_cols[0]]
        for i in range(1, len(C_cols)):
            if C_cols[i] - C_cols[i - 1] <= 2:
                current_group.append(C_cols[i])
            else:
                groups.append(current_group)
                current_group = [C_cols[i]]
        groups.append(current_group)
    # Prepare output
    out = [[0] * 7 for _ in range(8)]
    out[0][3] = 3
    # Collect h_events: dict row -> is_right
    h_events = {}
    prev_end_row = 0
    for g_idx, group in enumerate(groups):
        if g_idx == 0:
            start_row = 1
        else:
            prev_last_col = groups[g_idx - 1][-1]
            diff = group[0] - prev_last_col
            inc = 1 if diff <= 4 else diff - 3
            start_row = prev_end_row + inc
        # Direction for this group
        is_first_group = (g_idx == 0)
        start_dir_right = not (is_first_group and C_cols[0] == 0)
        for local_i in range(len(group)):
            this_row = start_row + local_i
            if this_row > 7:
                break
            is_right = start_dir_right if (local_i % 2 == 0) else not start_dir_right
            h_events[this_row] = is_right
        prev_end_row = start_row + len(group) - 1
    # Now simulate placement
    head_col = 3
    for r in range(1, 8):
        if r in h_events:
            is_right = h_events[r]
            if is_right:
                if head_col + 1 < 7:  # assume valid
                    out[r][head_col] = 2
                    out[r][head_col + 1] = 2
                head_col += 1
            else:
                if head_col - 1 >= 0:
                    out[r][head_col - 1] = 2
                    out[r][head_col] = 2
                head_col -= 1
        else:
            if 0 <= head_col < 7:
                out[r][head_col] = 2
            # head unchanged
    return out