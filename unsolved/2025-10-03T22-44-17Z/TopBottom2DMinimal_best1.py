def transform(grid: list[list[int]]) -> list[list[int]]:
    if not grid or not grid[0]:
        return []
    h = len(grid)
    w = len(grid[0])
    # Find all green positions to determine borders
    green_positions = [(r, c) for r in range(h) for c in range(w) if grid[r][c] == 3]
    if not green_positions:
        return [row[:] for row in grid]  # No change if no greens
    top_border_row = min(r for r, c in green_positions)
    left_border_col = min(c for r, c in green_positions)
    
    # Compute vertical shift
    top_over_rows = sum(1 for rr in range(top_border_row) if any(grid[rr][cc] == 2 for cc in range(w)))
    has_red_top = any(grid[top_border_row][cc] == 2 for cc in range(w))
    v_shift = top_over_rows + (1 if has_red_top else 0)
    
    # Compute horizontal shift
    left_over_cols = sum(1 for cc in range(left_border_col) if any(grid[rr][cc] == 2 for rr in range(h)))
    has_red_left = any(grid[rr][left_border_col] == 2 for rr in range(h))
    h_shift = left_over_cols + (1 if has_red_left else 0)
    
    # Create output: keep greens as 3, set everything else to 0
    output = [[3 if grid[r][c] == 3 else 0 for c in range(w)] for r in range(h)]
    
    # Move the 2's
    for r in range(h):
        for c in range(w):
            if grid[r][c] == 2:
                new_r = r + v_shift
                new_c = c + h_shift
                if 0 <= new_r < h and 0 <= new_c < w:
                    output[new_r][new_c] = 2
    
    return output