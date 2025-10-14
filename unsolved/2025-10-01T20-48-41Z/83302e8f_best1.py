from collections import deque
from typing import List

def transform(grid: List[List[int]]) -> List[List[int]]:
    if not grid or not grid[0]:
        return grid
    h = len(grid)
    w = len(grid[0])
    out = [row[:] for row in grid]
    
    # Find line_color
    non_zeros = set()
    for row in grid:
        for val in row:
            if val != 0:
                non_zeros.add(val)
    if not non_zeros:
        return out  # all zeros, no change
    line_color = next(iter(non_zeros))  # assume single non-zero color
    
    # Determine outer and inner colors
    if line_color <= 4:
        outer_color = 3
        inner_color = 4
    else:
        if line_color % 2 == 0:
            outer_color = 3
            inner_color = 4
        else:
            outer_color = 4
            inner_color = 3
    
    visited = [[False] * w for _ in range(h)]
    
    # Directions for 4-connectivity
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    
    def flood(start_r: int, start_c: int, color: int) -> None:
        if not (0 <= start_r < h and 0 <= start_c < w) or out[start_r][start_c] != 0 or visited[start_r][start_c]:
            return
        q = deque([(start_r, start_c)])
        visited[start_r][start_c] = True
        out[start_r][start_c] = color
        while q:
            r, c = q.popleft()
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and out[nr][nc] == 0 and not visited[nr][nc]:
                    visited[nr][nc] = True
                    out[nr][nc] = color
                    q.append((nr, nc))
    
    # Flood outer: start from all border 0 cells
    border_q = deque()
    # Left and right borders
    for i in range(h):
        if out[i][0] == 0 and not visited[i][0]:
            visited[i][0] = True
            out[i][0] = outer_color
            border_q.append((i, 0))
        if w > 1 and out[i][w-1] == 0 and not visited[i][w-1]:
            visited[i][w-1] = True
            out[i][w-1] = outer_color
            border_q.append((i, w-1))
    # Top and bottom borders, excluding corners already handled
    for j in range(w):
        if out[0][j] == 0 and not visited[0][j]:
            visited[0][j] = True
            out[0][j] = outer_color
            border_q.append((0, j))
        if h > 1 and out[h-1][j] == 0 and not visited[h-1][j]:
            visited[h-1][j] = True
            out[h-1][j] = outer_color
            border_q.append((h-1, j))
    
    # BFS for outer components
    while border_q:
        r, c = border_q.popleft()
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and out[nr][nc] == 0 and not visited[nr][nc]:
                visited[nr][nc] = True
                out[nr][nc] = outer_color
                border_q.append((nr, nc))
    
    # Now flood remaining internal 0's with inner_color
    for i in range(h):
        for j in range(w):
            if out[i][j] == 0:  # implies not visited
                flood(i, j, inner_color)
    
    return out