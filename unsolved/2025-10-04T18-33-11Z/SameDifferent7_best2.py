def transform(grid: list[list[int]]) -> list[list[int]]:
    # Flatten the grid
    flat = []
    for row in grid:
        flat.extend(row)
    
    # Extract non-zero values
    non_zeros = [x for x in flat if x != 0]
    
    # Create a new flat list of zeros of the same length
    new_flat = [0] * len(flat)
    
    # Place non-zero values at positions that are triangular numbers (0, 1, 3, 6, 10, ...)
    # but only as many as we have non_zeros
    for idx in range(len(non_zeros)):
        pos = idx * (idx + 1) // 2  # the idx-th triangular number
        if pos >= len(new_flat):
            break
        new_flat[pos] = non_zeros[idx]
    
    # Reshape back to original grid shape
    rows = len(grid)
    cols = len(grid[0])
    new_grid = []
    index = 0
    for i in range(rows):
        row = []
        for j in range(cols):
            row.append(new_flat[index])
            index += 1
        new_grid.append(row)
    
    return new_grid