def transform(grid: list[list[int]]) -> list[list[int]]:
    # Extract left pattern to determine the digit
    left_pattern = []
    for r in range(1, 4):
        row = []
        for c in range(1, 4):
            row.append(1 if grid[r][c] == 5 else 0)
        left_pattern.append(row)
    
    pattern_key = tuple(tuple(row) for row in left_pattern)
    
    digit_map = {
        ((1, 1, 1), (0, 1, 0), (0, 1, 0)): 1,
        ((1, 1, 0), (0, 1, 0), (0, 1, 1)): 2,
        ((0, 1, 1), (0, 1, 0), (1, 1, 0)): 3,
    }
    
    d = digit_map.get(pattern_key)
    if d is None:
        # Fallback, though not needed for given examples
        d = 0
    
    # Extract right pattern and transform
    output = []
    for r in range(1, 4):
        row = []
        for c in range(5, 8):
            cell = grid[r][c]
            row.append(d if cell == 5 else 0)
        output.append(row)
    
    return output