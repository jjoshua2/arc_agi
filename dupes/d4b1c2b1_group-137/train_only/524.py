from collections import deque

def transform(grid_lst):
    if not grid_lst or not grid_lst[0]:
        return grid_lst
    
    grid = [row[:] for row in grid_lst]
    rows = len(grid)
    cols = len(grid[0])
    
    # Find all seed positions (cells with value 1)
    seeds = []
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == 1:
                seeds.append((i, j))
    
    # If no seeds, return original grid
    if not seeds:
        return grid
    
    # BFS to propagate from seeds
    queue = deque(seeds)
    visited = set(seeds)
    
    while queue:
        i, j = queue.popleft()
        
        # Check all four directions
        for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            ni, nj = i + di, j + dj
            
            # Check bounds
            if 0 <= ni < rows and 0 <= nj < cols:
                # Only process empty cells (0) that haven't been visited
                if grid[ni][nj] == 0 and (ni, nj) not in visited:
                    grid[ni][nj] = 1
                    visited.add((ni, nj))
                    queue.append((ni, nj))
    
    return grid