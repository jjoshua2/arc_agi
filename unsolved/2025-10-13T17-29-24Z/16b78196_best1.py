from typing import List

def transform(grid: List[List[int]]) -> List[List[int]]:
    if not grid:
        return []
    rows = len(grid)
    output = [[0, 0] for _ in range(rows)]
    i = 0
    while i < rows:
        if all(cell == 0 for cell in grid[i]):
            # black row
            i += 1
            continue
        # start of band, take 2 rows
        band_rows = [grid[i], grid[i + 1]]
        # extract panels 2x2
        left = [[band_rows[0][0], band_rows[0][1]], [band_rows[1][0], band_rows[1][1]]]
        middle = [[band_rows[0][3], band_rows[0][4]], [band_rows[1][3], band_rows[1][4]]]
        right = [[band_rows[0][7], band_rows[0][8]], [band_rows[1][7], band_rows[1][8]]]
        panels = [left, middle, right]
        # sets of unique colors in each panel
        panel_sets = [set(sum(panel, [])) for panel in panels]
        sets = panel_sets
        # find outlier
        outlier_idx = -1
        if sets[0] == sets[1] and sets[0] != sets[2]:
            outlier_idx = 2
        elif sets[0] == sets[2] and sets[0] != sets[1]:
            outlier_idx = 1
        elif sets[1] == sets[2] and sets[1] != sets[0]:
            outlier_idx = 0
        # assume found
        outlier = panels[outlier_idx]
        r0 = outlier[0]
        r1 = outlier[1]
        # check if both columns uniform
        col0_uniform = (r0[0] == r1[0])
        col1_uniform = (r0[1] == r1[1])
        if col0_uniform and col1_uniform:
            # case both columns uniform
            c0 = r0[0]
            c1 = r0[1]
            small, large = min(c0, c1), max(c0, c1)
            output[i] = [small, small]
            output[i + 1] = [large, large]
        elif (r0[0] == r0[1]) and (r1[0] == r1[1]) and (r0[0] != r1[0]):
            # case both rows uniform, different colors
            c0 = r0[0]
            c1 = r1[0]
            small, large = min(c0, c1), max(c0, c1)
            output[i] = [small, large]
            output[i + 1] = [small, large]
        else:
            # one row uniform, one different
            if r0[0] == r0[1]:
                # top uniform, bottom different
                U = r0[0]
                D1, D2 = r1[0], r1[1]
                diff_is_top = False
            else:
                # bottom uniform, top different
                U = r1[0]
                D1, D2 = r0[0], r0[1]
                diff_is_top = True
            # determine common side
            if D1 == U:
                common_side = 'left'
                diff_color = D2
            else:
                common_side = 'right'
                diff_color = D1
            p_idx = outlier_idx
            pair_top = [0, 0]
            pair_bottom = [0, 0]
            if diff_is_top:
                if p_idx == 2 and common_side == 'left':
                    # sort ascending top
                    small = min(U, diff_color)
                    large = max(U, diff_color)
                    pair_top = [small, large]
                    pair_bottom = [U, U]
                elif p_idx == 1 and common_side == 'left':
                    # set top uniform, bottom original
                    pair_top = [U, U]
                    pair_bottom = [D1, D2]
                elif p_idx == 1 and common_side == 'right':
                    # set both to original top
                    pair_top = [D1, D2]
                    pair_bottom = [D1, D2]
                # other cases assumed not to occur
            else:
                # diff bottom
                if p_idx == 2 and common_side == 'right':
                    # sort descending bottom
                    large = max(U, diff_color)
                    small = min(U, diff_color)
                    pair_bottom = [large, small]
                    pair_top = [U, U]
                # other cases assumed not to occur
            output[i] = pair_top
            output[i + 1] = pair_bottom
        i += 2
    return output