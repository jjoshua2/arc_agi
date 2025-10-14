from typing import List, Tuple
from collections import defaultdict

def transform(grid: List[List[int]]) -> List[List[int]]:
    if not grid or not grid[0]:
        return [row[:] for row in grid]
    
    rows = len(grid)
    cols = len(grid[0])
    
    # Find all positions with 8
    eight_positions = [(i, j) for i in range(rows) for j in range(cols) if grid[i][j] == 8]
    if not eight_positions:
        return [row[:] for row in grid]
    
    # Bounding box of original 8s
    min_r = min(i for i, j in eight_positions)
    max_r = max(i for i, j in eight_positions)
    min_c = min(j for i, j in eight_positions)
    max_c = max(j for i, j in eight_positions)
    
    h = max_r - min_r + 1
    w = max_c - min_c + 1
    
    # Create pattern: True where there was 8
    pattern = [[False] * w for _ in range(h)]
    for i in range(h):
        for j in range(w):
            if grid[min_r + i][min_c + j] == 8:
                pattern[i][j] = True
    
    # Find all matching positions for replication
    matches = []
    for sr in range(rows - h + 1):
        for sc in range(cols - w + 1):
            is_match = True
            for i in range(h):
                for j in range(w):
                    required = 1 if pattern[i][j] else 0
                    if grid[sr + i][sc + j] != required:
                        is_match = False
                        break
                if not is_match:
                    break
            if is_match:
                matches.append((sr, sc))
    
    # Create output grid
    output = [row[:] for row in grid]
    
    # Perform replications: set 8s in matching positions
    for sr, sc in matches:
        for i in range(h):
            for j in range(w):
                if pattern[i][j]:
                    output[sr + i][sc + j] = 8
    
    # All locations: original + matches
    all_locations = [(min_r, min_c)] + matches
    
    # Horizontal fills: group by starting row sr
    horiz_groups = defaultdict(list)
    for sr, sc in all_locations:
        horiz_groups[sr].append(sc)
    
    for sr, sc_list in horiz_groups.items():
        if len(sc_list) > 1:
            sc_list = sorted(sc_list)
            for k in range(len(sc_list) - 1):
                left_sc = sc_list[k]
                right_sc = sc_list[k + 1]
                gap_start_col = left_sc + w
                gap_end_col = right_sc  # range up to right_sc - 1
                for c in range(gap_start_col, gap_end_col):
                    if c >= cols:
                        break
                    for rr in range(sr, sr + h):
                        if output[rr][c] == 1:
                            output[rr][c] = 7
    
    # Vertical fills: group by starting column sc
    vert_groups = defaultdict(list)
    for sr, sc in all_locations:
        vert_groups[sc].append(sr)
    
    for sc, sr_list in vert_groups.items():
        if len(sr_list) > 1:
            sr_list = sorted(sr_list)
            for k in range(len(sr_list) - 1):
                top_sr = sr_list[k]
                bot_sr = sr_list[k + 1]
                gap_start_row = top_sr + h
                gap_end_row = bot_sr  # range up to bot_sr - 1
                for rr in range(gap_start_row, gap_end_row):
                    if rr >= rows:
                        break
                    for cc in range(sc, sc + w):
                        if output[rr][cc] == 1:
                            output[rr][cc] = 7
    
    return output