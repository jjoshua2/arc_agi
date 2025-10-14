def transform(input_grid: list[list[int]]) -> list[list[int]]:
    if not input_grid or not input_grid[0]:
        return []
    rows = len(input_grid)
    cols = len(input_grid[0])
    grid = [row[:] for row in input_grid]  # Copy the grid
    
    max_zeros = -1
    best_r, best_c = -1, -1
    
    # Find the 5x5 subgrid with the maximum number of 0s
    for r in range(rows - 4):
        for c in range(cols - 4):
            zero_count = 0
            for i in range(5):
                for j in range(5):
                    if input_grid[r + i][c + j] == 0:
                        zero_count += 1
            if zero_count > max_zeros:
                max_zeros = zero_count
                best_r = r
                best_c = c
    
    if best_r == -1 or max_zeros == -1:
        return input_grid  # No change if no valid subgrid
    
    # Fill the border with 5
    sr, sc = best_r, best_c
    size = 5
    
    # Top border
    for j in range(size):
        grid[sr][sc + j] = 5
    # Bottom border
    for j in range(size):
        grid[sr + size - 1][sc + j] = 5
    # Left border (excluding corners)
    for i in range(1, size - 1):
        grid[sr + i][sc] = 5
    # Right border (excluding corners)
    for i in range(1, size - 1):
        grid[sr + i][sc + size - 1] = 5
    
    return grid