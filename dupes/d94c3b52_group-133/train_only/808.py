from typing import List, Tuple
from collections import defaultdict

def find_eight_positions(grid: List[List[int]]) -> List[Tuple[int, int]]:
    positions = []
    rows = len(grid)
    cols = len(grid[0])
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == 8:
                positions.append((i, j))
    return positions

def get_bounding_box(positions: List[Tuple[int, int]]) -> Tuple[int, int, int, int]:
    if not positions:
        return 0, 0, 0, 0
    min_r = min(i for i, j in positions)
    max_r = max(i for i, j in positions)
    min_c = min(j for i, j in positions)
    max_c = max(j for i, j in positions)
    return min_r, max_r, min_c, max_c

def transform(grid_lst: List[List[int]]) -> List[List[int]]:
    if not grid_lst or not grid_lst[0]:
        return [row[:] for row in grid_lst]
    
    grid = [row[:] for row in grid_lst]
    rows = len(grid)
    cols = len(grid[0])
    
    eight_positions = find_eight_positions(grid)
    if not eight_positions:
        return [row[:] for row in grid]
    
    min_r, max_r, min_c, max_c = get_bounding_box(eight_positions)
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
                gap_end_col = right_sc
                for c in range(gap_start_col, gap_end_col):
                    if c >= cols:
                        break
                    for rr in range(sr, sr + h):
                        if rr < rows and output[rr][c] == 1:
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
                gap_end_row = bot_sr
                for rr in range(gap_start_row, gap_end_row):
                    if rr >= rows:
                        break
                    for c in range(sc, sc + w):
                        if c < cols and output[rr][c] == 1:
                            output[rr][c] = 7
    
    return output