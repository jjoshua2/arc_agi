from collections import deque
from typing import List, Tuple

def transform(grid: List[List[int]]) -> List[List[int]]:
    if not grid or not grid[0]:
        return []
    h = len(grid)
    w = len(grid[0])
    
    # Directions for 4-connectivity
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    # Step 1: Find all base components (colors != 0 and != 1) and map shapes to colors
    shape_to_c: dict[frozenset[Tuple[int, int]], int] = {}
    visited = [[False] * w for _ in range(h)]
    
    for i in range(h):
        for j in range(w):
            if not visited[i][j] and grid[i][j] not in {0, 1}:
                # BFS to find connected component of same color
                c = grid[i][j]
                positions: List[Tuple[int, int]] = []
                q = deque([(i, j)])
                visited[i][j] = True
                while q:
                    x, y = q.popleft()
                    positions.append((x, y))
                    for dx, dy in directions:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < h and 0 <= ny < w and not visited[nx][ny] and grid[nx][ny] == c:
                            visited[nx][ny] = True
                            q.append((nx, ny))
                
                if positions:
                    min_r = min(p[0] for p in positions)
                    min_c = min(p[1] for p in positions)
                    rel = frozenset((p[0] - min_r, p[1] - min_c) for p in positions)
                    # Assume no conflicts; if conflict, last one wins
                    shape_to_c[rel] = c
    
    # Step 2: Copy grid to output
    out = [row[:] for row in grid]
    
    # Step 3: Find all 1-components and recolor if matching shape
    visited = [[False] * w for _ in range(h)]
    
    for i in range(h):
        for j in range(w):
            if not visited[i][j] and grid[i][j] == 1:
                # BFS to find connected component of 1's
                positions: List[Tuple[int, int]] = []
                q = deque([(i, j)])
                visited[i][j] = True
                while q:
                    x, y = q.popleft()
                    positions.append((x, y))
                    for dx, dy in directions:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < h and 0 <= ny < w and not visited[nx][ny] and grid[nx][ny] == 1:
                            visited[nx][ny] = True
                            q.append((nx, ny))
                
                if positions:
                    min_r = min(p[0] for p in positions)
                    min_c = min(p[1] for p in positions)
                    rel = frozenset((p[0] - min_r, p[1] - min_c) for p in positions)
                    if rel in shape_to_c:
                        c = shape_to_c[rel]
                        for px, py in positions:
                            out[px][py] = c
    
    return out