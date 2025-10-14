import numpy as np

def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    grid = np.array(grid_lst)
    rows, cols = grid.shape
    
    # Create output copy
    output = grid.copy()
    
    # Find non-zero regions
    non_zero_mask = grid != 0
    if not np.any(non_zero_mask):
        return output.tolist()
    
    # Get bounding box of non-zero region
    non_zero_coords = np.argwhere(non_zero_mask)
    min_row, min_col = np.min(non_zero_coords, axis=0)
    max_row, max_col = np.max(non_zero_coords, axis=0)
    
    # Check left and right borders for patterns
    left_border_colors = set()
    right_border_colors = set()
    
    for r in range(rows):
        if grid[r, 0] != 0:
            left_border_colors.add(grid[r, 0])
        if grid[r, cols-1] != 0:
            right_border_colors.add(grid[r, cols-1])
    
    # Check top and bottom borders for patterns
    top_border_colors = set()
    bottom_border_colors = set()
    
    for c in range(cols):
        if grid[0, c] != 0:
            top_border_colors.add(grid[0, c])
        if grid[rows-1, c] != 0:
            bottom_border_colors.add(grid[rows-1, c])
    
    # For left border colors, extend rightward through the non-zero region
    for color in left_border_colors:
        for r in range(rows):
            if grid[r, 0] == color:
                # Find how far this color extends horizontally
                c = 0
                while c < cols and grid[r, c] == color:
                    c += 1
                # If it extends at least 1 cell, create a horizontal channel
                if c > 1:
                    for channel_c in range(min_col, max_col + 1):
                        output[r, channel_c] = color
    
    # For right border colors, extend leftward through the non-zero region
    for color in right_border_colors:
        for r in range(rows):
            if grid[r, cols-1] == color:
                # Find how far this color extends horizontally
                c = cols - 1
                while c >= 0 and grid[r, c] == color:
                    c -= 1
                # If it extends at least 1 cell, create a horizontal channel
                if c < cols - 1:
                    for channel_c in range(min_col, max_col + 1):
                        output[r, channel_c] = color
    
    # For top border colors, extend downward through the non-zero region
    for color in top_border_colors:
        for c in range(cols):
            if grid[0, c] == color:
                # Find how far this color extends vertically
                r = 0
                while r < rows and grid[r, c] == color:
                    r += 1
                # If it extends at least 1 cell, create a vertical channel
                if r > 1:
                    for channel_r in range(min_row, max_row + 1):
                        output[channel_r, c] = color
    
    # For bottom border colors, extend upward through the non-zero region
    for color in bottom_border_colors:
        for c in range(cols):
            if grid[rows-1, c] == color:
                # Find how far this color extends vertically
                r = rows - 1
                while r >= 0 and grid[r, c] == color:
                    r -= 1
                # If it extends at least 1 cell, create a vertical channel
                if r < rows - 1:
                    for channel_r in range(min_row, max_row + 1):
                        output[channel_r, c] = color
    
    return output.tolist()