def transform(grid: list[list[int]]) -> list[list[int]]:
    if not grid or len(grid) != 3 or len(grid[0]) != 3:
        return grid  # Assuming always 3x3, but safety
    
    # Original
    orig = [row[:] for row in grid]
    
    # Horizontal flip: reverse each row
    flipped = [row[::-1] for row in grid]
    
    # Now, build the output rows
    output = []
    for i in range(3):
        row = (flipped[i] + orig[i] + flipped[i] + orig[i])
        output.append(row)
    
    return output