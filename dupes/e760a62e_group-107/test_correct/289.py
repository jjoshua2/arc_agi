from collections import defaultdict
import copy

def transform(grid: list[list[int]]) -> list[list[int]]:
    if not grid or not grid[0]:
        return grid
    rows = len(grid)
    cols = len(grid[0])
    
    # Find separator rows (full rows of 8)
    sep_rows = [i for i in range(rows) if all(grid[i][j] == 8 for j in range(cols))]
    
    # Find separator columns (full columns of 8)
    sep_cols = [j for j in range(cols) if all(grid[i][j] == 8 for i in range(rows))]
    
    # Row blocks: list of (start, end) inclusive
    row_blocks = []
    prev = -1
    for sr in sep_rows + [rows]:
        if prev + 1 <= sr - 1:
            row_blocks.append((prev + 1, sr - 1))
        prev = sr
    num_rb = len(row_blocks)
    
    # Col blocks
    col_blocks = []
    prev = -1
    for sc in sep_cols + [cols]:
        if prev + 1 <= sc - 1:
            col_blocks.append((prev + 1, sc - 1))
        prev = sc
    num_cb = len(col_blocks)
    
    # Horizontal fills
    horiz_fill = [[0] * num_cb for _ in range(num_rb)]
    for rb in range(num_rb):
        rs, re = row_blocks[rb]
        color_to_cbs = defaultdict(list)
        for cb in range(num_cb):
            cs, ce = col_blocks[cb]
            colors = set()
            for r in range(rs, re + 1):
                for c in range(cs, ce + 1):
                    val = grid[r][c]
                    if val != 0 and val != 8:
                        colors.add(val)
            if len(colors) == 1:
                colr = next(iter(colors))
                color_to_cbs[colr].append(cb)
        for colr, cblist in color_to_cbs.items():
            if cblist:
                min_cb = min(cblist)
                max_cb = max(cblist)
                for cb in range(min_cb, max_cb + 1):
                    horiz_fill[rb][cb] = colr
    
    # Vertical fills
    vert_fill = [[0] * num_cb for _ in range(num_rb)]
    for cb in range(num_cb):
        cs, ce = col_blocks[cb]
        color_to_rbs = defaultdict(list)
        for rb in range(num_rb):
            rs, re = row_blocks[rb]
            colors = set()
            for c in range(cs, ce + 1):
                for r in range(rs, re + 1):
                    val = grid[r][c]
                    if val != 0 and val != 8:
                        colors.add(val)
            if len(colors) == 1:
                colr = next(iter(colors))
                color_to_rbs[colr].append(rb)
        for colr, rblist in color_to_rbs.items():
            if rblist:
                min_rb = min(rblist)
                max_rb = max(rblist)
                for rb in range(min_rb, max_rb + 1):
                    vert_fill[rb][cb] = colr
    
    # Create output
    output = copy.deepcopy(grid)
    for rb in range(num_rb):
        rs, re = row_blocks[rb]
        for cb in range(num_cb):
            cs, ce = col_blocks[cb]
            hcol = horiz_fill[rb][cb]
            vcol = vert_fill[rb][cb]
            if hcol == 0 and vcol == 0:
                continue
            if hcol and vcol:
                if hcol == vcol:
                    final = hcol
                elif set([hcol, vcol]) == {2, 3}:
                    final = 6
                else:
                    final = max(hcol, vcol)  # Arbitrary for unknown cases
            elif hcol:
                final = hcol
            else:
                final = vcol
            # Fill only 0 cells
            for r in range(rs, re + 1):
                for c in range(cs, ce + 1):
                    if output[r][c] == 0:
                        output[r][c] = final
    return output