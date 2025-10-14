import numpy as np

def transform(grid_lst):
    if not grid_lst or not grid_lst[0]:
        return grid_lst
    
    grid = np.array(grid_lst)
    rows, cols = grid.shape
    
    # Find all rectangular regions with solid borders
    rectangles = []
    for t in range(rows):
        for b in range(t + 2, rows):  # Need at least 1 row inside
            for l in range(cols):
                for r in range(l + 2, cols):  # Need at least 1 column inside
                    # Check if all borders are the same color and non-zero
                    border_color = grid[t, l]
                    if border_color == 0:
                        continue
                    
                    # Check top border
                    if not np.all(grid[t, l:r+1] == border_color):
                        continue
                    
                    # Check bottom border
                    if not np.all(grid[b, l:r+1] == border_color):
                        continue
                    
                    # Check left border
                    if not np.all(grid[t:b+1, l] == border_color):
                        continue
                    
                    # Check right border
                    if not np.all(grid[t:b+1, r] == border_color):
                        continue
                    
                    rectangles.append((t, b, l, r, border_color))
    
    # Process each rectangle
    output = grid.copy()
    for t, b, l, r, color in rectangles:
        if color in [5, 7]:  # Remove grey and orange rectangles
            output[t:b+1, l:r+1] = 0
        elif color in [3, 2, 4, 6, 1]:  # Fill with pattern
            # Create a maze-like pattern inside
            inside_h = b - t - 1
            inside_w = r - l - 1
            
            if inside_h > 0 and inside_w > 0:
                # Create a pattern based on the position/size
                pattern = np.zeros((inside_h, inside_w), dtype=int)
                
                # Fill with alternating pattern
                for i in range(inside_h):
                    for j in range(inside_w):
                        # Simple alternating pattern based on position
                        if (i + j) % 2 == 0:
                            pattern[i, j] = color
                        else:
                            pattern[i, j] = 0
                
                output[t+1:b, l+1:r] = pattern
    
    return output.tolist()