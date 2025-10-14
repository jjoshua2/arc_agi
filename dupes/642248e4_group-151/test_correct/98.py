def transform(grid: list[list[int]]) -> list[list[int]]:
    if not grid or not grid[0]:
        return []
    
    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0
    
    if rows == 0 or cols == 0:
        return []
    
    # Check top row uniform
    top_color = None
    row0 = grid[0]
    first_top = row0[0]
    if first_top != 0 and all(x == first_top for x in row0):
        top_color = first_top
    
    # Check bottom row uniform
    bottom_color = None
    row_last = grid[-1]
    first_bottom = row_last[0]
    if first_bottom != 0 and all(x == first_bottom for x in row_last):
        bottom_color = first_bottom
    
    # Check left column uniform
    left_color = None
    first_left = grid[0][0]
    if first_left != 0 and all(grid[i][0] == first_left for i in range(rows)):
        left_color = first_left
    
    # Check right column uniform
    right_color = None
    first_right = grid[0][cols - 1]
    if first_right != 0 and all(grid[i][cols - 1] == first_right for i in range(rows)):
        right_color = first_right
    
    mode = None
    if top_color is not None and bottom_color is not None and top_color != bottom_color:
        mode = 'vertical'
        border_top = top_color
        border_bottom = bottom_color
    elif left_color is not None and right_color is not None and left_color != right_color:
        mode = 'horizontal'
        border_left = left_color
        border_right = right_color
    
    output = [row[:] for row in grid]
    
    if mode is None:
        return output
    
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == 1:
                if mode == 'vertical':
                    dist_top = i
                    dist_bottom = rows - 1 - i
                    if dist_top < dist_bottom:
                        if i > 0 and output[i - 1][j] == 0:
                            output[i - 1][j] = border_top
                    elif dist_top > dist_bottom:
                        if i < rows - 1 and output[i + 1][j] == 0:
                            output[i + 1][j] = border_bottom
                else:  # horizontal
                    dist_left = j
                    dist_right = cols - 1 - j
                    if dist_left < dist_right:
                        if j > 0 and output[i][j - 1] == 0:
                            output[i][j - 1] = border_left
                    elif dist_left > dist_right:
                        if j < cols - 1 and output[i][j + 1] == 0:
                            output[i][j + 1] = border_right
    
    return output