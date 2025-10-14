import numpy as np

def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    grid = np.array(grid_lst)
    h, w = grid.shape
    out = grid.copy().astype(int)
    center_r = h // 2
    center_c = w // 2

    # Process horizontal bars for side fills (n >= 3)
    for r in range(h):
        c = 0
        while c < w:
            if grid[r, c] == 4:
                start = c
                while c < w and grid[r, c] == 4:
                    c += 1
                end = c - 1
                n = end - start + 1
                if n >= 3:
                    is_lower = r > center_r
                    inner_r = r - 1 if is_lower else r + 1
                    outer_r = r + 1 if is_lower else r - 1
                    outer_side = 'lower' if is_lower else 'upper'
                    parity3 = start % 2
                    # Find attachments
                    has_attachment = False
                    attached_cols = []
                    attached_side = None
                    for cc in range(start, end + 1):
                        upper_att = (r > 0 and grid[r - 1, cc] == 4)
                        lower_att = (r < h - 1 and grid[r + 1, cc] == 4)
                        if upper_att or lower_att:
                            has_attachment = True
                            attached_cols.append(cc)
                            if upper_att and attached_side != 'lower':
                                attached_side = 'upper'
                            if lower_att and attached_side != 'upper':
                                attached_side = 'lower'
                    # Inner fill
                    if 0 <= inner_r < h:
                        c_fill = start if (start % 2 == parity3) else (start + 1 if start + 1 <= end else -1)
                        while c_fill <= end and c_fill >= 0:
                            skip = False
                            if has_attachment and attached_side == outer_side:
                                for att_c in attached_cols:
                                    if c_fill < att_c:
                                        skip = True
                                        break
                            if not skip and out[inner_r, c_fill] == 0:
                                out[inner_r, c_fill] = 3
                            c_fill += 2
                    # Outer regular fill
                    do_regular_outer = not has_attachment
                    if do_regular_outer and 0 <= outer_r < h:
                        parity1 = 1 - parity3
                        c_fill = start if (start % 2 == parity1) else (start + 1 if start + 1 <= end else -1)
                        while c_fill <= end and c_fill >= 0:
                            if out[outer_r, c_fill] == 0:
                                out[outer_r, c_fill] = 1
                            c_fill += 2
                    # Special for attached on outer_r
                    if has_attachment and 0 <= outer_r < h:
                        for att_c in attached_cols:
                            if out[outer_r, att_c] == 0:
                                out[outer_r, att_c] = 1
            else:
                c += 1

    # Process vertical bars for side fills (n >= 3)
    for c in range(w):
        r = 0
        while r < h:
            if grid[r, c] == 4:
                start = r
                while r < h and grid[r, c] == 4:
                    r += 1
                end = r - 1
                n = end - start + 1
                if n >= 3:
                    is_right = c > center_c
                    inner_c = c - 1 if is_right else c + 1
                    outer_c = c + 1 if is_right else c - 1
                    outer_dir = 'right' if is_right else 'left'
                    parity3 = start % 2
                    # Find attachments
                    has_attachment = False
                    attached_rows = []
                    attached_dirs = set()
                    for rr in range(start, end + 1):
                        left_att = (c > 0 and grid[rr, c - 1] == 4)
                        right_att = (c < w - 1 and grid[rr, c + 1] == 4)
                        if left_att or right_att:
                            has_attachment = True
                            attached_rows.append(rr)
                            if left_att:
                                attached_dirs.add('left')
                            if right_att:
                                attached_dirs.add('right')
                    # Special for attached_top
                    attached_top = start in attached_rows
                    # Inner fill
                    if 0 <= inner_c < w:
                        r_fill = start if (start % 2 == parity3) else (start + 1 if start + 1 <= end else -1)
                        while r_fill <= end and r_fill >= 0:
                            skip = False
                            color = 3
                            if has_attachment and min(attached_rows) != start:  # not only top
                                if r_fill > max(attached_rows):
                                    skip = True
                            if attached_top and r_fill == start:
                                color = 1
                            if not skip and out[r_fill, inner_c] == 0:
                                out[r_fill, inner_c] = color
                            r_fill += 2
                    # Outer regular fill
                    do_regular_outer = not has_attachment
                    if do_regular_outer and 0 <= outer_c < w:
                        parity1 = 1 - parity3
                        r_fill = start if (start % 2 == parity1) else (start + 1 if start + 1 <= end else -1)
                        while r_fill <= end and r_fill >= 0:
                            if out[r_fill, outer_c] == 0:
                                out[r_fill, outer_c] = 1
                            r_fill += 2
                    # Special for attached on outer_c
                    if has_attachment and 0 <= outer_c < w:
                        for att_r in attached_rows:
                            if out[att_r, outer_c] == 0:
                                out[att_r, outer_c] = 1
                    # Special extension for attached_top
                    if attached_top:
                        # special color already handled in inner fill
                        # extension at bottom
                        if 'right' in attached_dirs:
                            # count beyond
                            beyond = 0
                            tc = c + 1
                            ar = start  # attached row
                            while tc < w and grid[ar, tc] == 4:
                                beyond += 1
                                tc += 1
                            # extend beyond cells right from c at end row
                            for i in range(beyond):
                                tc = c + 1 + i
                                if tc < w and out[end, tc] == 0:
                                    out[end, tc] = 3
                        if 'left' in attached_dirs:
                            # extend 1 cell left from c at end row
                            tc = c - 1
                            if tc >= 0 and out[end, tc] == 0:
                                out[end, tc] = 3
            else:
                r += 1

    # End extensions for all bars n >=1 horizontal and vertical

    # Horizontal end extensions
    for r in range(h):
        c = 0
        while c < w:
            if grid[r, c] == 4:
                start = c
                while c < w and grid[r, c] == 4:
                    c += 1
                end = c - 1
                n = end - start + 1
                if n >= 1:
                    # left end
                    if start > 0 and out[r, start - 1] == 0:
                        block = False
                        if r > 0 and grid[r - 1, start] == 2:
                            block = True
                        if r < h - 1 and grid[r + 1, start] == 2:
                            block = True
                        if not block:
                            out[r, start - 1] = 1
                    # right end
                    if end < w - 1 and out[r, end + 1] == 0:
                        block = False
                        if r > 0 and grid[r - 1, end] == 2:
                            block = True
                        if r < h - 1 and grid[r + 1, end] == 2:
                            block = True
                        if not block:
                            out[r, end + 1] = 1
            else:
                c += 1

    # Vertical end extensions
    for c in range(w):
        r = 0
        while r < h:
            if grid[r, c] == 4:
                start = r
                while r < h and grid[r, c] == 4:
                    r += 1
                end = r - 1
                n = end - start + 1
                if n >= 1:
                    has_attached_top = False
                    if n >= 1 and ( (c > 0 and grid[start, c - 1] == 4) or (c < w - 1 and grid[start, c + 1] == 4) ):
                        has_attached_top = True
                    # top end
                    if start > 0 and out[start - 1, c] == 0:
                        block = False
                        if c > 0 and grid[start, c - 1] == 2:
                            block = True
                        if c < w - 1 and grid[start, c + 1] == 2:
                            block = True
                        if not block:
                            out[start - 1, c] = 1
                    # bottom end
                    skip_bottom = has_attached_top
                    if not skip_bottom and end < h - 1 and out[end + 1, c] == 0:
                        block = False
                        if c > 0 and grid[end, c - 1] == 2:
                            block = True
                        if c < w - 1 and grid[end, c + 1] == 2:
                            block = True
                        if not block:
                            out[end + 1, c] = 1
            else:
                r += 1

    return out.tolist()