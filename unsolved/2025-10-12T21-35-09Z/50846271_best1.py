import numpy as np

def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    grid = np.array(grid_lst)
    rows, cols = grid.shape
    output = grid.copy()

    # Gap filling for horizontal
    for r in range(rows):
        red_cols = np.where(grid[r] == 2)[0]
        if len(red_cols) < 2:
            continue
        segments = []
        i = 0
        while i < len(red_cols):
            start = red_cols[i]
            end = start
            i += 1
            while i < len(red_cols) and red_cols[i] == end + 1:
                end = red_cols[i]
                i += 1
            segments.append((start, end))
        for j in range(len(segments) - 1):
            s1_start, s1_end = segments[j]
            s2_start, s2_end = segments[j + 1]
            gap_size = s2_start - s1_end - 1
            if 0 < gap_size <= 2:
                for cc in range(s1_end + 1, s2_start):
                    output[r, cc] = 8

    # Gap filling for vertical
    for c in range(cols):
        red_rows = np.where(grid[:, c] == 2)[0]
        if len(red_rows) < 2:
            continue
        segments = []
        i = 0
        while i < len(red_rows):
            start = red_rows[i]
            end = start
            i += 1
            while i < len(red_rows) and red_rows[i] == end + 1:
                end = red_rows[i]
                i += 1
            segments.append((start, end))
        for j in range(len(segments) - 1):
            s1_start, s1_end = segments[j]
            s2_start, s2_end = segments[j + 1]
            gap_size = s2_start - s1_end - 1
            if 0 < gap_size <= 2:
                for rr in range(s1_end + 1, s2_start):
                    output[rr, c] = 8

    # Find horiz and vert bars
    def find_bars(out):
        h_bars = []
        for r in range(rows):
            i = 0
            while i < cols:
                if out[r, i] in [2, 8]:
                    start = i
                    while i < cols and out[r, i] in [2, 8]:
                        i += 1
                    h_bars.append((r, start, i - 1))
                else:
                    i += 1
        v_bars = []
        for c in range(cols):
            i = 0
            while i < rows:
                if out[i, c] in [2, 8]:
                    start = i
                    while i < rows and out[i, c] in [2, 8]:
                        i += 1
                    v_bars.append((c, start, i - 1))
                else:
                    i += 1
        return h_bars, v_bars

    horiz_bars, vert_bars = find_bars(output)

    # Collect intersections
    intersections = set()
    for hr, hstart, hend in horiz_bars:
        for vc, vstart, vend in vert_bars:
            r = hr
            c = vc
            if hstart <= c <= hend and vstart <= r <= vend:
                intersections.add((r, c))

    # First phase: balance arms for each intersection
    for r, c in intersections:
        # balance vertical
        # find current vstart vend by scanning
        vstart = r
        while vstart > 0 and output[vstart - 1, c] in [2, 8]:
            vstart -= 1
        vend = r
        while vend < rows - 1 and output[vend + 1, c] in [2, 8]:
            vend += 1
        up_arm = r - vstart
        down_arm = vend - r
        if up_arm < down_arm:
            extend_amount = down_arm - up_arm
            current = vstart - 1
            added = 0
            while added < extend_amount and current >= 0 and output[current, c] == 5:
                output[current, c] = 8
                added += 1
                current -= 1
        elif down_arm < up_arm:
            extend_amount = up_arm - down_arm
            current = vend + 1
            added = 0
            while added < extend_amount and current < rows and output[current, c] == 5:
                output[current, c] = 8
                added += 1
                current += 1

        # balance horizontal
        hstart = c
        while hstart > 0 and output[r, hstart - 1] in [2, 8]:
            hstart -= 1
        hend = c
        while hend < cols - 1 and output[r, hend + 1] in [2, 8]:
            hend += 1
        left_arm = c - hstart
        right_arm = hend - c
        if left_arm < right_arm:
            extend_amount = right_arm - left_arm
            current = hstart - 1
            added = 0
            while added < extend_amount and current >= 0 and output[r, current] == 5:
                output[r, current] = 8
                added += 1
                current -= 1
        elif right_arm < left_arm:
            extend_amount = left_arm - right_arm
            current = hend + 1
            added = 0
            while added < extend_amount and current < cols and output[r, current] == 5:
                output[r, current] = 8
                added += 1
                current += 1

    # Second phase
    for r, c in intersections:
        # compute current arms
        up_arm = 0
        rr = r - 1
        while rr >= 0 and output[rr, c] in [2, 8]:
            up_arm += 1
            rr -= 1
        down_arm = 0
        rr = r + 1
        while rr < rows and output[rr, c] in [2, 8]:
            down_arm += 1
            rr += 1
        left_arm = 0
        cc = c - 1
        while cc >= 0 and output[r, cc] in [2, 8]:
            left_arm += 1
            cc -= 1
        right_arm = 0
        cc = c + 1
        while cc < cols and output[r, cc] in [2, 8]:
            right_arm += 1
            cc += 1

        vert_max = max(up_arm, down_arm)
        horiz_max = max(left_arm, right_arm)

        if horiz_max < vert_max:
            target = vert_max
            # extend left
            if left_arm < target:
                extend_amount = target - left_arm
                cc = c - 1 - left_arm
                for _ in range(extend_amount):
                    if cc < 0 or output[r, cc] != 5:
                        break
                    output[r, cc] = 8
                    cc -= 1
            # extend right
            if right_arm < target:
                extend_amount = target - right_arm
                cc = c + 1 + right_arm
                for _ in range(extend_amount):
                    if cc >= cols or output[r, cc] != 5:
                        break
                    output[r, cc] = 8
                    cc += 1

        if vert_max < horiz_max:
            target = horiz_max
            # extend up
            if up_arm < target:
                extend_amount = target - up_arm
                rr = r - 1 - up_arm
                for _ in range(extend_amount):
                    if rr < 0 or output[rr, c] != 5:
                        break
                    output[rr, c] = 8
                    rr -= 1
            # extend down
            if down_arm < target:
                extend_amount = target - down_arm
                rr = r + 1 + down_arm
                for _ in range(extend_amount):
                    if rr >= rows or output[rr, c] != 5:
                        break
                    output[rr, c] = 8
                    rr += 1

    return output.tolist()