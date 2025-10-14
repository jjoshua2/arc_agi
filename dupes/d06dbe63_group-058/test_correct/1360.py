def transform(grid: list[list[int]]) -> list[list[int]]:
    if not grid:
        return []
    rows = len(grid)
    cols = len(grid[0]) if grid else 0
    output = [row[:] for row in grid]
    
    # Find the position of the purple cell (8)
    p_r, p_c = -1, -1
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 8:
                p_r, p_c = r, c
                break
        if p_r != -1:
            break
    
    if p_r == -1:
        return output  # No purple, no change
    
    for row in range(rows):
        if row == p_r:
            continue
        rel_row = row - p_r
        dist = abs(rel_row)
        if rel_row < 0:
            sign = 1
        else:
            sign = -1
        shift = sign * (dist - 1)
        if dist % 2 == 1:  # odd distance: single cell
            target_col = p_c + shift
            if 0 <= target_col < cols:
                output[row][target_col] = 5
        else:  # even distance: three cells centered
            center_col = p_c + shift
            for dc in [-1, 0, 1]:
                target_col = center_col + dc
                if 0 <= target_col < cols:
                    output[row][target_col] = 5
    
    return output