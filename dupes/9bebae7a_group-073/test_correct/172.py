from typing import List, Tuple

def transform(grid: List[List[int]]) -> List[List[int]]:
    if not grid or not grid[0]:
        return []
    rows = len(grid)
    cols = len(grid[0])
    output = [row[:] for row in grid]
    
    # Find yellow positions (4)
    yellow_pos: List[Tuple[int, int]] = [(i, j) for i in range(rows) for j in range(cols) if grid[i][j] == 4]
    if not yellow_pos:
        # Erase pink if any
        for i in range(rows):
            for j in range(cols):
                if grid[i][j] == 6:
                    output[i][j] = 0
        return output
    
    yellow_min_r = min(r for r, c in yellow_pos)
    yellow_max_r = max(r for r, c in yellow_pos)
    yellow_min_c = min(c for r, c in yellow_pos)
    yellow_max_c = max(c for r, c in yellow_pos)
    orig_num_r = yellow_max_r - yellow_min_r + 1
    orig_num_c = yellow_max_c - yellow_min_c + 1
    
    # Find pink positions (6)
    pink_pos: List[Tuple[int, int]] = [(i, j) for i in range(rows) for j in range(cols) if grid[i][j] == 6]
    if pink_pos:
        # Erase pink
        for r, c in pink_pos:
            output[r][c] = 0
        if pink_pos:
            pink_min_r = min(r for r, c in pink_pos)
            pink_max_r = max(r for r, c in pink_pos)
            pink_min_c = min(c for r, c in pink_pos)
            pink_max_c = max(c for r, c in pink_pos)
        else:
            # No pink, no change beyond erase (already done)
            return output
    else:
        # No pink, no change
        return output
    
    # Compute column overlap
    overlap_start = max(yellow_min_c, pink_min_c)
    overlap_end = min(yellow_max_c, pink_max_c)
    col_overlap = max(0, overlap_end - overlap_start + 1) if overlap_start <= overlap_end else 0
    
    if col_overlap > 0:
        # Vertical doubling
        needed_r = orig_num_r
        avail_up = yellow_min_r
        avail_down = rows - yellow_max_r - 1
        new_min_r = -1
        new_max_r = -1
        if needed_r <= avail_up:
            # Extend up
            new_min_r = yellow_min_r - needed_r
            new_max_r = yellow_max_r
        elif needed_r <= avail_down:
            # Extend down
            new_min_r = yellow_min_r
            new_max_r = yellow_max_r + needed_r
        if new_min_r == -1:
            # Cannot extend vertically, return as is
            return output
        
        center_r = (new_min_r + new_max_r) / 2.0
        for r, c in yellow_pos:
            new_r_float = 2 * center_r - r
            new_r = int(new_r_float)
            if 0 <= new_r < rows and output[new_r][c] == 0:
                output[new_r][c] = 4
    else:
        # Horizontal doubling
        needed_c = orig_num_c
        avail_right = cols - yellow_max_c - 1
        avail_left = yellow_min_c
        new_min_c = -1
        new_max_c = -1
        if needed_c <= avail_right:
            # Extend right
            new_min_c = yellow_min_c
            new_max_c = yellow_max_c + needed_c
        elif needed_c <= avail_left:
            # Extend left
            new_max_c = yellow_max_c
            new_min_c = yellow_min_c - needed_c
        if new_min_c == -1:
            # Cannot extend horizontally, return as is
            return output
        
        center_c = (new_min_c + new_max_c) / 2.0
        for r, c in yellow_pos:
            new_c_float = 2 * center_c - c
            new_c = int(new_c_float)
            if 0 <= new_c < cols and output[r][new_c] == 0:
                output[r][new_c] = 4
    
    return output