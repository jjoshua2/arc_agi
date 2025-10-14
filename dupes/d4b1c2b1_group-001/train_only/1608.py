import numpy as np

def is_diamond_shape(grid, start_row, start_col, color):
    """Check if there's a diamond shape starting at (start_row, start_col)"""
    rows, cols = grid.shape
    
    # Check if we have at least 3 rows for a diamond
    if start_row + 2 >= rows:
        return False
    
    # Check top row - should have at least 3 consecutive cells of the same color
    top_length = 0
    for c in range(start_col, cols):
        if grid[start_row, c] == color:
            top_length += 1
        else:
            break
    
    if top_length < 3:
        return False
    
    # Check middle rows - should form diamond pattern
    for i in range(1, top_length // 2 + 1):
        if start_row + i >= rows:
            return False
        
        # Left side should have color at appropriate position
        left_pos = start_col + i
        if left_pos >= cols or grid[start_row + i, left_pos] != color:
            return False
        
        # Right side should have color at appropriate position
        right_pos = start_col + top_length - 1 - i
        if right_pos < 0 or grid[start_row + i, right_pos] != color:
            return False
    
    # Check bottom row - should match top row pattern
    if start_row + top_length // 2 >= rows:
        return False
    
    for c in range(start_col, start_col + top_length):
        if grid[start_row + top_length // 2, c] != color:
            return False
    
    return True

def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    grid = np.array(grid_lst)
    rows, cols = grid.shape
    
    # Create a copy to modify
    result = grid.copy()
    
    # Find all diamond shapes
    for r in range(rows):
        for c in range(cols):
            color = grid[r, c]
            if color != 0:  # Non-black cell
                # Check if this is the start of a diamond shape
                if is_diamond_shape(grid, r, c, color):
                    # Shift the diamond right by 1 column if possible
                    diamond_width = 0
                    while c + diamond_width < cols and grid[r, c + diamond_width] == color:
                        diamond_width += 1
                    
                    # Check if we can shift right (right side has zeros)
                    can_shift = True
                    for i in range(diamond_width):
                        if c + i + 1 >= cols or grid[r, c + i + 1] != 0:
                            can_shift = False
                            break
                    
                    if can_shift:
                        # Shift the entire diamond right by 1
                        for dr in range(diamond_width // 2 + 1):
                            for dc in range(diamond_width):
                                if r + dr < rows and c + dc < cols and grid[r + dr, c + dc] == color:
                                    # Clear original position
                                    result[r + dr, c + dc] = 0
                                    # Set new position
                                    if c + dc + 1 < cols:
                                        result[r + dr, c + dc + 1] = color
    
    return result.tolist()