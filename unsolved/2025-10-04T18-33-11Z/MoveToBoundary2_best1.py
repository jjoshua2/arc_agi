import copy

def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    grid = copy.deepcopy(grid_lst)
    rows = len(grid)
    cols = len(grid[0])
    
    # First, identify all vertical orange columns that should remain fixed
    fixed_columns = set()
    for c in range(cols):
        column_values = [grid[r][c] for r in range(rows)]
        # Check if this is a vertical orange column (all orange or mostly orange with some gaps)
        orange_count = sum(1 for v in column_values if v == 7)
        if orange_count > rows / 2:  # More than half of the column is orange
            fixed_columns.add(c)
    
    # Simulate orange cells falling down
    # We need to process from bottom to top to handle stacking correctly
    for c in range(cols):
        if c in fixed_columns:
            continue  # Skip fixed columns
            
        # Process each column from bottom to top
        for r in range(rows - 2, -1, -1):  # Start from second last row, go upward
            if grid[r][c] == 7:  # Found an orange cell
                current_r = r
                # Try to move it down as far as possible
                while current_r + 1 < rows and grid[current_r + 1][c] != 7:
                    grid[current_r + 1][c] = 7
                    grid[current_r][c] = 0
                    current_r += 1
    
    return grid