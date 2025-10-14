from collections import deque

def transform(grid: list[list[int]]) -> list[list[int]]:
    if not grid or not grid[0]:
        return []
    rows = len(grid)
    cols = len(grid[0])
    output = [row[:] for row in grid]
    
    # Directions for flood fill
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    # Queue for BFS
    queue = deque()
    
    # Mark all border 0s as starting points for flood fill
    for r in range(rows):
        # Top and bottom borders
        if output[r][0] == 0:
            queue.append((r, 0))
            output[r][0] = -1
        if output[r][cols - 1] == 0:
            queue.append((r, cols - 1))
            output[r][cols - 1] = -1
    
    for c in range(cols):
        # Left and right borders (avoid corners if already added)
        if output[0][c] == 0:
            queue.append((0, c))
            output[0][c] = -1
        if output[rows - 1][c] == 0:
            queue.append((rows - 1, c))
            output[rows - 1][c] = -1
    
    # Flood fill from borders: mark all reachable 0s as -1 (outside)
    while queue:
        r, c = queue.popleft()
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and output[nr][nc] == 0:
                output[nr][nc] = -1
                queue.append((nr, nc))
    
    # Now, set unreached 0s (interiors) to 8, reset -1 to 0, keep 1s as 1
    for r in range(rows):
        for c in range(cols):
            if output[r][c] == 0:
                output[r][c] = 8
            elif output[r][c] == -1:
                output[r][c] = 0
            # 1s remain 1
    
    return output