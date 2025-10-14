from collections import deque

def transform(grid: list[list[int]]) -> list[list[int]]:
    if not grid or not grid[0]:
        return grid
        
    rows = len(grid)
    cols = len(grid[0])
    
    # Create output grid as copy of input
    output = [row[:] for row in grid]
    
    # Mark all border 0's and connected 0's as visited (not holes)
    visited = [[False] * cols for _ in range(rows)]
    queue = deque()
    
    # Add border 0's to queue
    for r in range(rows):
        if grid[r][0] == 0:
            queue.append((r, 0))
            visited[r][0] = True
        if grid[r][cols-1] == 0:
            queue.append((r, cols-1))
            visited[r][cols-1] = True
            
    for c in range(cols):
        if grid[0][c] == 0:
            queue.append((0, c))
            visited[0][c] = True
        if grid[rows-1][c] == 0:
            queue.append((rows-1, c))
            visited[rows-1][c] = True
    
    # BFS to mark all 0's connected to border
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    while queue:
        r, c = queue.popleft()
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc] and grid[nr][nc] == 0:
                visited[nr][nc] = True
                queue.append((nr, nc))
    
    # Fill unvisited 0's (holes) with red (2)
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 0 and not visited[r][c]:
                output[r][c] = 2
                
    return output