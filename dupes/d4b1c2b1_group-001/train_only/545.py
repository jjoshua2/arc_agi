import numpy as np

def transform(grid_lst):
    grid = np.array(grid_lst)
    rows, cols = grid.shape
    
    # Create output grid as a copy of input
    output = grid.copy()
    
    # Find all 3x3 blocks by looking for patterns
    # A 3x3 block has non-zero values in all positions
    
    # First, identify the top-left corners of all 3x3 blocks
    block_corners = []
    for r in range(rows - 2):
        for c in range(cols - 2):
            # Check if this is a 3x3 block (all non-zero)
            if (grid[r, c] != 0 and grid[r, c+1] != 0 and grid[r, c+2] != 0 and
                grid[r+1, c] != 0 and grid[r+1, c+1] != 0 and grid[r+1, c+2] != 0 and
                grid[r+2, c] != 0 and grid[r+2, c+1] != 0 and grid[r+2, c+2] != 0):
                block_corners.append((r, c))
    
    # For each block, get its center color
    block_centers = {}
    for r, c in block_corners:
        center_color = grid[r+1, c+1]
        block_centers[(r, c)] = center_color
    
    # Fill horizontal gaps between blocks
    for r in range(rows):
        for c in range(cols - 1):
            # Check if this is a gap between two blocks horizontally
            if grid[r, c] != 0 and grid[r, c+1] == 0:
                # Find the block to the left
                left_block = None
                for br, bc in block_corners:
                    if br <= r <= br+2 and bc <= c <= bc+2:
                        left_block = (br, bc)
                        break
                
                # Find the block to the right
                right_block = None
                for br, bc in block_corners:
                    if br <= r <= br+2 and bc <= c+1 <= bc+2:
                        right_block = (br, bc)
                        break
                
                # If both blocks found, fill the gap with left block's center color
                if left_block and right_block:
                    fill_color = block_centers[left_block]
                    output[r, c+1] = fill_color
    
    # Fill vertical gaps between blocks
    for r in range(rows - 1):
        for c in range(cols):
            # Check if this is a gap between two blocks vertically
            if grid[r, c] != 0 and grid[r+1, c] == 0:
                # Find the block above
                top_block = None
                for br, bc in block_corners:
                    if br <= r <= br+2 and bc <= c <= bc+2:
                        top_block = (br, bc)
                        break
                
                # Find the block below
                bottom_block = None
                for br, bc in block_corners:
                    if br <= r+1 <= br+2 and bc <= c <= bc+2:
                        bottom_block = (br, bc)
                        break
                
                # If both blocks found, fill the gap with top block's center color
                if top_block and bottom_block:
                    fill_color = block_centers[top_block]
                    output[r+1, c] = fill_color
    
    return output.tolist()