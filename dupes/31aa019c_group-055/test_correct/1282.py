def transform(grid: list[list[int]]) -> list[list[int]]:
    if not grid or not grid[0]:
        return []
    
    rows = len(grid)
    cols = len(grid[0])
    
    # Count frequency of each color
    count = [0] * 10
    for r in range(rows):
        for c in range(cols):
            color = grid[r][c]
            if 1 <= color <= 9:  # Ignore 0
                count[color] += 1
    
    # Find the color with exactly one occurrence
    unique_color = None
    for color in range(1, 10):
        if count[color] == 1:
            unique_color = color
            break
    
    if unique_color is None:
        # If no unique color, return all zeros? But assume there is one
        return [[0 for _ in range(cols)] for _ in range(rows)]
    
    # Find the position of the unique color
    pos_r, pos_c = -1, -1
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == unique_color:
                pos_r, pos_c = r, c
                break
        if pos_r != -1:
            break
    
    # Create output grid all zeros
    output = [[0 for _ in range(cols)] for _ in range(rows)]
    
    # Fill the 3x3 centered at (pos_r, pos_c) with 2's, center with unique_color
    for i in [-1, 0, 1]:
        for j in [-1, 0, 1]:
            nr = pos_r + i
            nc = pos_c + j
            if 0 <= nr < rows and 0 <= nc < cols:
                if i == 0 and j == 0:
                    output[nr][nc] = unique_color
                else:
                    output[nr][nc] = 2
    
    return output