def transform(grid: list[list[int]]) -> list[list[int]]:
    n = len(grid)
    output = [row[:] for row in grid]
    
    # Find top and bottom border rows
    top_border = grid[0]
    bottom_border = grid[n-1]
    
    # Determine middle row for top/bottom half division
    middle_row = n // 2
    
    for r in range(1, n-1):  # Skip border rows
        for c in range(n):
            if grid[r][c] != 0:  # Only process colored cells
                if r < middle_row:
                    # Top half - use top border color
                    output[r][c] = top_border[c]
                else:
                    # Bottom half - use bottom border color
                    output[r][c] = bottom_border[c]
    
    return output