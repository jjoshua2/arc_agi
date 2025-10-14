from collections import deque

def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    if not grid_lst or not grid_lst[0]:
        return []
    
    grid = [row[:] for row in grid_lst]
    rows, cols = len(grid), len(grid[0])
    
    visited = set()
    queue = deque()
    
    # Add all border 0s to queue and visited
    for i in range(rows):
        # left border
        if grid[i][0] == 0:
            queue.append((i, 0))
            visited.add((i, 0))
        # right border
        if grid[i][cols - 1] == 0:
            queue.append((i, cols - 1))
            visited.add((i, cols - 1))
    
    for j in range(cols):
        # top border
        if grid[0][j] == 0:
            queue.append((0, j))
            visited.add((0, j))
        # bottom border
        if grid[rows - 1][j] == 0:
            queue.append((rows - 1, j))
            visited.add((rows - 1, j))
    
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    
    while queue:
        x, y = queue.popleft()
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < rows and 0 <= ny < cols and grid[nx][ny] == 0 and (nx, ny) not in visited:
                visited.add((nx, ny))
                queue.append((nx, ny))
    
    # Fill unvisited 0s with 3
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == 0 and (i, j) not in visited:
                grid[i][j] = 3
    
    return grid