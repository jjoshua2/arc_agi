from collections import deque

def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    if not grid_lst or not grid_lst[0]:
        return []
    rows = len(grid_lst)
    cols = len(grid_lst[0])
    output = [row[:] for row in grid_lst]
    
    def is_open(val: int) -> bool:
        return val == 0
    
    visited = [[False for _ in range(cols)] for _ in range(rows)]
    
    # Directions: up, down, left, right
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    # BFS queue
    q = deque()
    
    # Add all border open cells to queue and mark visited
    # Top row
    for c in range(cols):
        if is_open(grid_lst[0][c]):
            q.append((0, c))
            visited[0][c] = True
    # Bottom row
    for c in range(cols):
        if is_open(grid_lst[rows - 1][c]):
            q.append((rows - 1, c))
            visited[rows - 1][c] = True
    # Left column, excluding corners already added
    for r in range(1, rows - 1):
        if is_open(grid_lst[r][0]):
            q.append((r, 0))
            visited[r][0] = True
    # Right column, excluding corners
    for r in range(1, rows - 1):
        if is_open(grid_lst[r][cols - 1]):
            q.append((r, cols - 1))
            visited[r][cols - 1] = True
    
    # Flood fill from border
    while q:
        r, c = q.popleft()
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc] and is_open(grid_lst[nr][nc]):
                visited[nr][nc] = True
                q.append((nr, nc))
    
    # Change enclosed open cells to 3
    for r in range(rows):
        for c in range(cols):
            if is_open(grid_lst[r][c]) and not visited[r][c]:
                output[r][c] = 3
    
    return output