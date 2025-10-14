import copy
import numpy as np

def find_squares(grid):
    rows = len(grid)
    cols = len(grid[0])
    visited = np.zeros((rows, cols), dtype=bool)
    squares = []
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] != 0 and not visited[i][j]:
                component = []
                stack = [(i, j)]
                color = grid[i][j]
                visited[i][j] = True
                while stack:
                    x, y = stack.pop()
                    component.append((x, y))
                    for dx, dy in directions:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < rows and 0 <= ny < cols and not visited[nx][ny] and grid[nx][ny] == color:
                            visited[nx][ny] = True
                            stack.append((nx, ny))
                
                if component:
                    min_r = min(p[0] for p in component)
                    max_r = max(p[0] for p in component)
                    min_c = min(p[1] for p in component)
                    max_c = max(p[1] for p in component)
                    height = max_r - min_r + 1
                    width = max_c - min_c + 1
                    if height == width and len(component) == height * width:
                        is_square = True
                        for rr in range(min_r, max_r + 1):
                            for cc in range(min_c, max_c + 1):
                                if grid[rr][cc] != color:
                                    is_square = False
                                    break
                            if not is_square:
                                break
                        if is_square:
                            squares.append((color, min_r, min_c, height))
    return squares

def transform(grid_lst):
    if not grid_lst:
        return []
    grid = [row[:] for row in grid_lst]
    rows = len(grid)
    cols_num = len(grid[0]) if rows > 0 else 0  # renamed to avoid conflict with cols var
    squares = find_squares(grid)
    
    output_grid = [row[:] for row in grid]
    
    # Horizontal pairs
    n = len(squares)
    for ii in range(n):
        for kk in range(ii + 1, n):
            color1, r1, c1, s1 = squares[ii]
            color2, r2, c2, s2 = squares[kk]
            if s1 == s2 and r1 == r2 and c2 == c1 + s1:
                # left square1, right square2
                main_is_left = color1 > color2
                if main_is_left:
                    main_color = color1
                    main_r = r1
                    main_c_start = c1
                    main_s = s1
                    attach_c_start = c2
                    direction = 'left'
                else:
                    main_color = color2
                    main_r = r2
                    main_c_start = c2
                    main_s = s2
                    attach_c_start = c1
                    direction = 'right'
                
                # horizontal mirror fill
                fill_r_start = main_r
                fill_r_end = main_r + main_s - 1
                if direction == 'left':
                    if main_s == 1:
                        amount = 1
                    else:
                        amount = main_s
                    fill_c_start = main_c_start - amount
                    fill_c_end = main_c_start - 1
                else:
                    if main_s == 1:
                        amount = 2
                    else:
                        amount = main_s
                    fill_c_start = main_c_start + main_s
                    fill_c_end = main_c_start + main_s + amount - 1
                
                for rr in range(fill_r_start, fill_r_end + 1):
                    for cc in range(max(0, fill_c_start), min(cols_num, fill_c_end + 1)):
                        if output_grid[rr][cc] == 0:
                            output_grid[rr][cc] = main_color
                
                # perpendicular columns
                if main_s == 1:
                    if direction == 'left':
                        perp_c = attach_c_start
                    else:
                        perp_c = main_c_start + main_s + amount - 1  # new end
                    perp_cs = [perp_c]
                else:
                    perp_cs = list(range(main_c_start, main_c_start + main_s))
                
                # up
                u_r_start = main_r - main_s
                u_r_end = main_r - 1
                for rr in range(max(0, u_r_start), min(rows, u_r_end + 1)):
                    for cc in perp_cs:
                        if 0 <= cc < cols_num and output_grid[rr][cc] == 0:
                            output_grid[rr][cc] = main_color
                
                # down
                d_r_start = main_r + main_s
                d_r_end = main_r + 2 * main_s - 1
                for rr in range(max(0, d_r_start), min(rows, d_r_end + 1)):
                    for cc in perp_cs:
                        if 0 <= cc < cols_num and output_grid[rr][cc] == 0:
                            output_grid[rr][cc] = main_color
    
    # Vertical pairs
    for ii in range(n):
        for kk in range(ii + 1, n):
            color1, r1, c1, s1 = squares[ii]
            color2, r2, c2, s2 = squares[kk]
            if s1 == s2 and c1 == c2 and r2 == r1 + s1:
                # upper square1, lower square2
                if color1 > color2:
                    main_color = color1
                    main_r = r1  # not used
                    attach_r = r2
                    c_start = c1
                    s = s1
                    is_purple = (main_color == 8)
                    
                    # mirror down
                    l_r_start = attach_r + s
                    l_r_end = attach_r + 2 * s - 1
                    l_c_start = c_start
                    l_c_end = c_start + s - 1
                    for rr in range(max(0, l_r_start), min(rows, l_r_end + 1)):
                        for cc in range(max(0, l_c_start), min(cols_num, l_c_end + 1)):
                            if output_grid[rr][cc] == 0:
                                output_grid[rr][cc] = main_color
                    
                    if is_purple:
                        # set middle lower to 0
                        for rr in range(max(0, l_r_start), min(rows, l_r_end + 1)):
                            for cc in range(max(0, l_c_start), min(cols_num, l_c_end + 1)):
                                output_grid[rr][cc] = 0
                        # add sides to lower
                        left_amount = s
                        l_fill_c_start = c_start - left_amount
                        l_fill_c_end = c_start - 1
                        for rr in range(max(0, l_r_start), min(rows, l_r_end + 1)):
                            for cc in range(max(0, l_fill_c_start), min(cols_num, l_fill_c_end + 1)):
                                if output_grid[rr][cc] == 0:
                                    output_grid[rr][cc] = main_color
                        r_fill_c_start = c_start + s
                        r_fill_c_end = c_start + s + left_amount - 1
                        for rr in range(max(0, l_r_start), min(rows, l_r_end + 1)):
                            for cc in range(max(0, r_fill_c_start), min(cols_num, r_fill_c_end + 1)):
                                if output_grid[rr][cc] == 0:
                                    output_grid[rr][cc] = main_color
                    
                    # horizontal for attachment rows
                    a_r_start = attach_r
                    a_r_end = attach_r + s - 1
                    if is_purple:
                        left_amount = s
                        right_amount = s
                    else:
                        left_amount = 2 * s
                        right_amount = 0
                    # left
                    l_fill_c_start = c_start - left_amount
                    l_fill_c_end = c_start - 1
                    for rr in range(max(0, a_r_start), min(rows, a_r_end + 1)):
                        for cc in range(max(0, l_fill_c_start), min(cols_num, l_fill_c_end + 1)):
                            if output_grid[rr][cc] == 0:
                                output_grid[rr][cc] = main_color
                    # right
                    r_fill_c_start = c_start + s
                    r_fill_c_end = c_start + s + right_amount - 1
                    for rr in range(max(0, a_r_start), min(rows, a_r_end + 1)):
                        for cc in range(max(0, r_fill_c_start), min(cols_num, r_fill_c_end + 1)):
                            if output_grid[rr][cc] == 0:
                                output_grid[rr][cc] = main_color
                    
                    # upper horizontal
                    u_r_start = r1
                    u_r_end = r1 + s - 1
                    if is_purple:
                        left_amount = s
                        right_amount = s
                        # left
                        l_fill_c_start = c_start - left_amount
                        l_fill_c_end = c_start - 1
                        for rr in range(max(0, u_r_start), min(rows, u_r_end + 1)):
                            for cc in range(max(0, l_fill_c_start), min(cols_num, l_fill_c_end + 1)):
                                if output_grid[rr][cc] == 0:
                                    output_grid[rr][cc] = main_color
                        # right
                        r_fill_c_start = c_start + s
                        r_fill_c_end = c_start + s + right_amount - 1
                        for rr in range(max(0, u_r_start), min(rows, u_r_end + 1)):
                            for cc in range(max(0, r_fill_c_start), min(cols_num, r_fill_c_end + 1)):
                                if output_grid[rr][cc] == 0:
                                    output_grid[rr][cc] = main_color
    
    return output_grid