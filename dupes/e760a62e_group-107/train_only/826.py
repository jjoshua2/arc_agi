from collections import defaultdict
import copy

def transform(grid: list[list[int]]) -> list[list[int]]:
    if not grid or not grid[0]:
        return grid
    rows_num = len(grid)
    cols_num = len(grid[0])
    output = [row[:] for row in grid]
    
    # Find sep_rows: rows fully 8
    sep_rows = [r for r in range(rows_num) if all(grid[r][c] == 8 for c in range(cols_num))]
    
    # Find sep_cols: columns fully 8
    sep_cols = [c for c in range(cols_num) if all(grid[r][c] == 8 for r in range(rows_num))]
    
    # Get row_blocks: list of (start, end) inclusive for open rows between seps
    row_blocks = []
    prev = -1
    for sr in sep_rows + [rows_num]:
        if prev + 1 < sr:
            row_blocks.append((prev + 1, sr - 1))
        prev = sr
    num_rb = len(row_blocks)
    
    # Get col_blocks
    col_blocks = []
    prev = -1
    for sc in sep_cols + [cols_num]:
        if prev + 1 < sc:
            col_blocks.append((prev + 1, sc - 1))
        prev = sc
    num_cb = len(col_blocks)
    
    # Collect seeds: (rb_id, cb_id) -> C
    seeds = {}
    vertical_seeds = defaultdict(lambda: defaultdict(list))
    for rb_id in range(num_rb):
        rs, re = row_blocks[rb_id]
        for cb_id in range(num_cb):
            cs, ce = col_blocks[cb_id]
            colors = set()
            for r in range(rs, re + 1):
                for c in range(cs, ce + 1):
                    val = grid[r][c]
                    if val != 0 and val != 8:
                        colors.add(val)
            if len(colors) == 1:
                C = next(iter(colors))
                seeds[(rb_id, cb_id)] = C
                vertical_seeds[cb_id][C].append(rb_id)
    
    # Vertical fills
    vertical_fill = [[0] * num_cb for _ in range(num_rb)]
    for cb_id in vertical_seeds:
        for C, rb_list in vertical_seeds[cb_id].items():
            if rb_list:
                min_rb = min(rb_list)
                max_rb = max(rb_list)
                for rb_id in range(min_rb, max_rb + 1):
                    vertical_fill[rb_id][cb_id] = C
    
    # Horizontal seeds: rb_id -> list of cb_id
    horizontal_seeds = defaultdict(list)
    for (rb_id, cb_id), C in seeds.items():
        horizontal_seeds[rb_id].append(cb_id)
    
    # Horizontal fills
    horizontal_fill = [[0] * num_cb for _ in range(num_rb)]
    for rb_id in horizontal_seeds:
        cb_list = horizontal_seeds[rb_id]
        if cb_list:
            colors = {seeds[(rb_id, cb)] for cb in cb_list}
            if len(colors) == 1:
                C = next(iter(colors))
                min_cb = min(cb_list)
                max_cb = max(cb_list)
                for cb_id in range(min_cb, max_cb + 1):
                    horizontal_fill[rb_id][cb_id] = C
    
    # Apply fills
    for rb_id in range(num_rb):
        rs, re = row_blocks[rb_id]
        for cb_id in range(num_cb):
            cs, ce = col_blocks[cb_id]
            v = vertical_fill[rb_id][cb_id]
            h = horizontal_fill[rb_id][cb_id]
            if v and h:
                if v == h:
                    final = v
                else:
                    final = 6
            elif v:
                final = v
            elif h:
                final = h
            else:
                final = 0
            # Fill 0s only
            for r in range(rs, re + 1):
                for c in range(cs, ce + 1):
                    if output[r][c] == 0:
                        output[r][c] = final
    
    return output