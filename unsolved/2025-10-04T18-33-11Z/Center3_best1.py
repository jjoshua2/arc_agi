def transform(grid: list[list[int]]) -> list[list[int]]:
    if not grid or not grid[0]:
        return []
    
    rows = len(grid)
    cols = len(grid[0])
    
    # Find all distinct non-zero colors
    colors_set = set()
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 0:
                colors_set.add(grid[r][c])
    
    if not colors_set:
        # If no colors, perhaps return empty or original, but examples have colors
        return []
    
    # Sort the colors
    sorted_colors = sorted(colors_set)
    # Take the middle one (index len//2)
    selected_color = sorted_colors[len(sorted_colors) // 2]
    
    # Find all positions with selected_color
    positions = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == selected_color:
                positions.append((r, c))
    
    if not positions:
        return []
    
    # Find min and max r, c
    min_r = min(pos[0] for pos in positions)
    max_r = max(pos[0] for pos in positions)
    min_c = min(pos[1] for pos in positions)
    max_c = max(pos[1] for pos in positions)
    
    # Create the output grid
    height = max_r - min_r + 1
    width = max_c - min_c + 1
    output = [[0 for _ in range(width)] for _ in range(height)]
    
    # Fill the output
    for i in range(height):
        for j in range(width):
            orig_r = min_r + i
            orig_c = min_c + j
            if grid[orig_r][orig_c] == selected_color:
                output[i][j] = selected_color
            else:
                output[i][j] = 0
    
    return output