def transform(grid: list[list[int]]) -> list[list[int]]:
    if not grid:
        return []
    h = len(grid)
    w = len(grid[0]) if h > 0 else 0
    # Find the 5
    r5 = -1
    c5 = -1
    for rr in range(h):
        for cc in range(w):
            if grid[rr][cc] == 5:
                r5 = rr
                c5 = cc
                break
        if r5 != -1:
            break
    if r5 == -1:
        # No 5, but assume present
        return [[0] * 7 for _ in range(h)]
    out_c = c5 - 8
    out_grid = [[0] * 7 for _ in range(h)]
    out_grid[r5][out_c] = 5
    current_r = r5
    current_c = out_c

    # Parse shapes
    left_shapes = []
    right_shapes = []
    i = 0
    while i < h - 2:
        # Check if block starts at i
        has_content = False
        for j in range(i, min(i + 3, h)):
            for k in [0, 1, 2, 4, 5, 6]:
                if 0 <= k < w and grid[j][k] != 0:
                    has_content = True
                    break
            if has_content:
                break
        if not has_content:
            i += 1
            continue

        # Left panel cols 0-2
        left_filled = set()
        left_color = -1
        for j in range(3):
            if i + j >= h:
                break
            for k in range(3):
                val = grid[i + j][k]
                if val != 0:
                    left_filled.add((j, k))
                    if left_color == -1:
                        left_color = val
        if len(left_filled) > 0 and left_color != -1:
            if left_filled == {(0, 0), (0, 1), (1, 0), (1, 2), (2, 1)}:
                left_shapes.append(('A', left_color))
            elif left_filled == {(0, 0), (0, 2), (1, 1), (2, 1)}:
                left_shapes.append(('B', left_color))
            elif left_filled == {(0, 0), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2)}:
                left_shapes.append(('D', left_color))
            elif left_filled == {(0, 0), (0, 1), (0, 2), (1, 1), (2, 0), (2, 2)}:
                left_shapes.append(('C', left_color))

        # Right panel cols 4-6
        right_filled = set()
        right_color = -1
        for j in range(3):
            if i + j >= h:
                break
            for k in range(3):
                col_idx = 4 + k
                if col_idx >= w:
                    continue
                val = grid[i + j][col_idx]
                if val != 0:
                    right_filled.add((j, k))
                    if right_color == -1:
                        right_color = val
        if len(right_filled) > 0 and right_color != -1:
            if right_filled == {(0, 0), (0, 1), (1, 0), (1, 2), (2, 1)}:
                right_shapes.append(('A', right_color))
            elif right_filled == {(0, 0), (0, 2), (1, 1), (2, 1)}:
                right_shapes.append(('B', right_color))
            elif right_filled == {(0, 0), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2)}:
                right_shapes.append(('D', right_color))
            elif right_filled == {(0, 0), (0, 1), (0, 2), (1, 1), (2, 0), (2, 2)}:
                right_shapes.append(('C', right_color))

        i += 3

    all_shapes = left_shapes + right_shapes

    # Place shapes
    for shape in all_shapes:
        typ, colr = shape
        if typ == 'A':
            direc = 'hor_right'
            leng = 3
        elif typ == 'B':
            direc = 'vert_down'
            leng = 2
        elif typ == 'D':
            direc = 'hor_left'
            leng = 2
        elif typ == 'C':
            direc = 'hor_left'
            leng = 4
        else:
            continue  # unknown

        next_r = current_r + 1
        next_c = current_c
        if direc == 'vert_down':
            for d in range(leng):
                out_r = next_r + d
                out_c_pos = next_c
                if 0 <= out_r < h and 0 <= out_c_pos < 7:
                    out_grid[out_r][out_c_pos] = colr
            current_r = next_r + leng - 1
            current_c = next_c
        elif direc == 'hor_right':
            for d in range(leng):
                out_r = next_r
                out_c_pos = next_c + d
                if 0 <= out_r < h and 0 <= out_c_pos < 7:
                    out_grid[out_r][out_c_pos] = colr
            current_r = next_r
            current_c = next_c + leng - 1
        elif direc == 'hor_left':
            for d in range(leng):
                out_r = next_r
                out_c_pos = next_c - d
                if 0 <= out_r < h and 0 <= out_c_pos < 7:
                    out_grid[out_r][out_c_pos] = colr
            current_r = next_r
            current_c = next_c - leng + 1

    return out_grid