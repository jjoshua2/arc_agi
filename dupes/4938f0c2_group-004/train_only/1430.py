from typing import List

def transform(grid: List[List[int]]) -> List[List[int]]:
    if not grid or not grid[0]:
        return []
    rows = len(grid)
    cols = len(grid[0])
    output = [row[:] for row in grid]
    
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right
    
    visited = set()
    queue = []  # Use list as queue
    
    # Add all border 0 cells to queue and visited
    for r in range(rows):
        # Left border
        if grid[r][0] == 0:
            queue.append((r, 0))
            visited.add((r, 0))
        # Right border
        if grid[r][cols - 1] == 0:
            queue.append((r, cols - 1))
            visited.add((r, cols - 1))
    
    for c in range(cols):
        # Top border
        if grid[0][c] == 0:
            queue.append((0, c))
            visited.add((0, c))
        # Bottom border
        if grid[rows - 1][c] == 0:
            queue.append((rows - 1, c))
            visited.add((rows - 1, c))
    
    # BFS to mark all reachable 0 cells
    while queue:
        cr, cc = queue.pop(0)  # pop(0) for FIFO
        for dr, dc in directions:
            nr, nc = cr + dr, cc + dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 0 and (nr, nc) not in visited:
                visited.add((nr, nc))
                queue.append((nr, nc))
    
    # Change unmarked 0 cells to 4 (interior voids)
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 0 and (r, c) not in visited:
                output[r][c] = 4
    
    return output