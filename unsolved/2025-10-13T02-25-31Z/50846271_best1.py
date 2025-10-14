import copy

def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    if not grid_lst or not grid_lst[0]:
        return []
    
    output = copy.deepcopy(grid_lst)
    rows = len(output)
    cols = len(output[0])
    
    while True:
        changed = False
        
        # Gap fill horizontal
        for r in range(rows):
            positions = [c for c in range(cols) if output[r][c] in (2, 8)]
            for j in range(len(positions) - 1):
                p = positions[j]
                q = positions[j + 1]
                for k in range(p + 1, q):
                    if output[r][k] == 5:
                        output[r][k] = 8
                        changed = True
        
        # Gap fill vertical
        for c in range(cols):
            positions = [r for r in range(rows) if output[r][c] in (2, 8)]
            for j in range(len(positions) - 1):
                p = positions[j]
                q = positions[j + 1]
                for k in range(p + 1, q):
                    if output[k][c] == 5:
                        output[k][c] = 8
                        changed = True
        
        # Find centers (only on 2)
        centers = []
        for i in range(rows):
            for j in range(cols):
                if output[i][j] == 2:
                    has_horiz = (j > 0 and output[i][j - 1] in (2, 8)) or (j < cols - 1 and output[i][j + 1] in (2, 8))
                    has_vert = (i > 0 and output[i - 1][j] in (2, 8)) or (i < rows - 1 and output[i + 1][j] in (2, 8))
                    if has_horiz and has_vert:
                        centers.append((i, j))
        
        # Extend arms for each center
        for ci, cj in centers:
            # Calculate arm lengths
            upper = 0
            x = ci - 1
            while x >= 0 and output[x][cj] in (2, 8):
                upper += 1
                x -= 1
            
            lower = 0
            x = ci + 1
            while x < rows and output[x][cj] in (2, 8):
                lower += 1
                x += 1
            
            left = 0
            y = cj - 1
            while y >= 0 and output[ci][y] in (2, 8):
                left += 1
                y -= 1
            
            right = 0
            y = cj + 1
            while y < cols and output[ci][y] in (2, 8):
                right += 1
                y += 1
            
            max_arm = max(upper, lower, left, right)
            
            # Directions config: (arm_len, start_ex, step, is_vertical)
            directions = [
                (upper, ci - 1 - upper, -1, True),
                (lower, ci + 1 + lower, 1, True),
                (left, cj - 1 - left, -1, False),
                (right, cj + 1 + right, 1, False)
            ]
            
            for arm_len, ex, step, is_vertical in directions:
                diff = max_arm - arm_len
                if diff > 0 and arm_len > 0:
                    count = 0
                    while count < diff:
                        if is_vertical:
                            if not (0 <= ex < rows):
                                break
                            if grid_lst[ex][cj] != 5:
                                break
                            output[ex][cj] = 8
                        else:
                            if not (0 <= ex < cols):
                                break
                            if grid_lst[ci][ex] != 5:
                                break
                            output[ci][ex] = 8
                        changed = True
                        count += 1
                        ex += step
        
        if not changed:
            break
    
    return output