from collections import deque

def transform(grid: list[list[int]]) -> list[list[int]]:
    if not grid or not grid[0]:
        return []
    rows = len(grid)
    cols = len(grid[0])
    if rows == 0 or cols == 0:
        return []
    
    marked = [[False] * cols for _ in range(rows)]
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    # Flood fill from border 0 cells
    # Left and right columns
    for r in range(rows):
        for c in [0, cols - 1]:
            if grid[r][c] == 0 and not marked[r][c]:
                q = deque([(r, c)])
                marked[r][c] = True
                while q:
                    cr, cc = q.popleft()
                    for dr, dc in directions:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < rows and 0 <= nc < cols and not marked[nr][nc] and grid[nr][nc] == 0:
                            marked[nr][nc] = True
                            q.append((nr, nc))
    
    # Top and bottom rows (corners already handled if 0)
    for c in range(cols):
        for r in [0, rows - 1]:
            if grid[r][c] == 0 and not marked[r][c]:
                q = deque([(r, c)])
                marked[r][c] = True
                while q:
                    cr, cc = q.popleft()
                    for dr, dc in directions:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < rows and 0 <= nc < cols and not marked[nr][nc] and grid[nr][nc] == 0:
                            marked[nr][nc] = True
                            q.append((nr, nc))
    
    # Create output by filling unmarked 0s with 8
    output = [row[:] for row in grid]
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 0 and not marked[r][c]:
                output[r][c] = 8
    return output