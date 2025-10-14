from collections import deque

def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    if not grid_lst or not grid_lst[0]:
        return []
    
    rows = len(grid_lst)
    cols = len(grid_lst[0])
    visited = [[False] * cols for _ in range(rows)]
    
    # Directions for 4-connected flood fill (up, down, left, right)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    # BFS starting from all border 0 cells
    q = deque()
    for r in range(rows):
        for c in range(cols):
            # Check if it's a border cell and is 0
            if (r == 0 or r == rows - 1 or c == 0 or c == cols - 1) and grid_lst[r][c] == 0:
                if not visited[r][c]:
                    q.append((r, c))
                    visited[r][c] = True
    
    # Flood fill all reachable 0 cells from border
    while q:
        r, c = q.popleft()
        for dr, dc in directions:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc] and grid_lst[nr][nc] == 0:
                visited[nr][nc] = True
                q.append((nr, nc))
    
    # Create output: copy input, but fill unvisited 0's with 3
    output = [[grid_lst[r][c] for c in range(cols)] for r in range(rows)]
    for r in range(rows):
        for c in range(cols):
            if grid_lst[r][c] == 0 and not visited[r][c]:
                output[r][c] = 3
    
    return output