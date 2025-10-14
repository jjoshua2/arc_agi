def transform(grid: list[list[int]]) -> list[list[int]]:
    # Extract left pattern (rows 1-3, cols 1-3) as binary (5 -> 1, else 0)
    left_pattern = []
    for r in range(1, 4):
        row = []
        for c in range(1, 4):
            row.append(1 if grid[r][c] == 5 else 0)
        left_pattern.append(tuple(row))
    pattern_key = tuple(left_pattern)
    
    # Digit mapping based on the patterns
    digit_map = {
        ((1, 1, 1), (0, 1, 0), (0, 1, 0)): 1,
        ((1, 1, 0), (0, 1, 0), (0, 1, 1)): 2,
        ((0, 1, 1), (0, 1, 0), (1, 1, 0)): 3,
    }
    
    d = digit_map.get(pattern_key, 0)  # Fallback to 0 if unknown, but shouldn't happen
    
    # Create 3x3 output based on right pattern (rows 1-3, cols 5-7)
    output = []
    for r in range(1, 4):
        row = []
        for c in range(5, 8):
            cell = grid[r][c]
            row.append(d if cell == 5 else 0)
        output.append(row)
    
    return output