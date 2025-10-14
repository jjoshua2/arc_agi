from collections import Counter, deque

def transform(grid: list[list[int]]) -> list[list[int]]:
    if not grid or not grid[0]:
        return grid
    
    rows = len(grid)
    cols = len(grid[0])
    
    # Find the most common non-zero color (wall color)
    counter = Counter()
    for r in range(rows):
        for c in range(cols):
            val = grid[r][c]
            if val != 0:
                counter[val] += 1
    
    if not counter:
        return [row[:] for row in grid]
    
    wall_color = counter.most_common(1)[0][0]
    
    # Mark outside regions: flood fill from borders through non-wall cells
    outside = [[False for _ in range(cols)] for _ in range(rows)]
    queue = deque()
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    
    # Add border non-wall cells to queue
    # Top row
    for c in range(cols):
        if grid[0][c] != wall_color:
            queue.append((0, c))
            outside[0][c] = True
    # Bottom row
    for c in range(cols):
        if grid[rows - 1][c] != wall_color:
            queue.append((rows - 1, c))
            outside[rows - 1][c] = True
    # Left column, excluding corners
    for r in range(1, rows - 1):
        if grid[r][0] != wall_color:
            queue.append((r, 0))
            outside[r][0] = True
    # Right column, excluding corners
    for r in range(1, rows - 1):
        if grid[r][cols - 1] != wall_color:
            queue.append((r, cols - 1))
            outside[r][cols - 1] = True
    
    # BFS
    while queue:
        cr, cc = queue.popleft()
        for dr, dc in directions:
            nr, nc = cr + dr, cc + dc
            if (0 <= nr < rows and 0 <= nc < cols and
                grid[nr][nc] != wall_color and not outside[nr][nc]):
                outside[nr][nc] = True
                queue.append((nr, nc))
    
    # Create output
    output = [row[:] for row in grid]
    for r in range(rows):
        for c in range(cols):
            if (grid[r][c] == wall_color or
                (grid[r][c] != 0 and not outside[r][c])):
                output[r][c] = 0
    
    return output