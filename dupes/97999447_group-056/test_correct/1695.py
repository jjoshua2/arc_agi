def transform(input_grid: list[list[int]]) -> list[list[int]]:
    # Create a copy of the input grid
    output = [row[:] for row in input_grid]
    
    rows = len(input_grid)
    cols = len(input_grid[0])
    
    # Find all colored cells (non-zero)
    colored_cells = []
    for r in range(rows):
        for c in range(cols):
            if input_grid[r][c] != 0:
                colored_cells.append((r, c, input_grid[r][c]))
    
    # For each colored cell, create alternating pattern to the right
    for r, start_c, color in colored_cells:
        # Start from the original column position
        current_col = start_c
        use_color = True  # Start with the original color
        
        while current_col < cols:
            if use_color:
                output[r][current_col] = color
            else:
                output[r][current_col] = 5  # Grey
            
            use_color = not use_color
            current_col += 1
    
    return output