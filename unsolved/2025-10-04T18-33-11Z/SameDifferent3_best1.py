def transform(grid: list[list[int]]) -> list[list[int]]:
    if not grid or not grid[0]:
        return grid
    
    rows = len(grid)
    cols = len(grid[0])
    
    # Count frequency of each color (excluding 0)
    from collections import defaultdict
    color_count = defaultdict(int)
    for r in range(rows):
        for c in range(cols):
            color = grid[r][c]
            if color != 0:
                color_count[color] += 1
    
    if not color_count:
        return [row[:] for row in grid]
    
    # Find the color with maximum frequency
    max_color = max(color_count.items(), key=lambda x: x[1])[0]
    
    # Create output grid
    output = []
    for r in range(rows):
        new_row = []
        for c in range(cols):
            color = grid[r][c]
            if color == 0 or color == max_color:
                new_row.append(color)
            else:
                new_row.append(0)
        output.append(new_row)
    
    return output