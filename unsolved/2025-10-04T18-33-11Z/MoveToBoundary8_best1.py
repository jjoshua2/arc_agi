import numpy as np

def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    grid = np.array(grid_lst)
    rows, cols = grid.shape
    
    # Find the longest contiguous stripe of non-zero, non-black color
    def find_longest_stripe(grid):
        max_length = 0
        stripe_color = 0
        stripe_positions = []
        stripe_orientation = None  # 'horizontal' or 'vertical'
        
        # Check horizontal stripes
        for r in range(rows):
            current_color = 0
            current_length = 0
            start_c = 0
            for c in range(cols):
                if grid[r, c] != 0 and grid[r, c] == current_color:
                    current_length += 1
                else:
                    if current_length > max_length and current_color != 0:
                        max_length = current_length
                        stripe_color = current_color
                        stripe_positions = [(r, sc) for sc in range(start_c, start_c + current_length)]
                        stripe_orientation = 'horizontal'
                    if grid[r, c] != 0:
                        current_color = grid[r, c]
                        current_length = 1
                        start_c = c
                    else:
                        current_color = 0
                        current_length = 0
            if current_length > max_length and current_color != 0:
                max_length = current_length
                stripe_color = current_color
                stripe_positions = [(r, sc) for sc in range(start_c, start_c + current_length)]
                stripe_orientation = 'horizontal'
        
        # Check vertical stripes
        for c in range(cols):
            current_color = 0
            current_length = 0
            start_r = 0
            for r in range(rows):
                if grid[r, c] != 0 and grid[r, c] == current_color:
                    current_length += 1
                else:
                    if current_length > max_length and current_color != 0:
                        max_length = current_length
                        stripe_color = current_color
                        stripe_positions = [(sr, c) for sr in range(start_r, start_r + current_length)]
                        stripe_orientation = 'vertical'
                    if grid[r, c] != 0:
                        current_color = grid[r, c]
                        current_length = 1
                        start_r = r
                    else:
                        current_color = 0
                        current_length = 0
            if current_length > max_length and current_color != 0:
                max_length = current_length
                stripe_color = current_color
                stripe_positions = [(sr, c) for sr in range(start_r, start_r + current_length)]
                stripe_orientation = 'vertical'
        
        return stripe_color, stripe_positions, stripe_orientation
    
    # Find moving cells (non-zero, non-black, not part of the stripe)
    def find_moving_cells(grid, stripe_positions):
        moving_cells = []
        for r in range(rows):
            for c in range(cols):
                if grid[r, c] != 0 and (r, c) not in stripe_positions:
                    moving_cells.append((r, c, grid[r, c]))
        return moving_cells
    
    stripe_color, stripe_positions, stripe_orientation = find_longest_stripe(grid)
    moving_cells = find_moving_cells(grid, stripe_positions)
    
    # Create output grid (same as input initially)
    output = grid.copy()
    
    if stripe_orientation == 'horizontal':
        # Horizontal stripe - move cells vertically toward it
        stripe_row = stripe_positions[0][0]
        for r, c, color in moving_cells:
            output[r, c] = 0  # Remove original cell
            new_r = stripe_row
            output[new_r, c] = color  # Move to stripe row
        
    elif stripe_orientation == 'vertical':
        # Vertical stripe - move cells horizontally toward it
        stripe_col = stripe_positions[0][1]
        for r, c, color in moving_cells:
            output[r, c] = 0  # Remove original cell
            new_c = stripe_col
            output[r, new_c] = color  # Move to stripe column
    
    return output.tolist()