from collections import deque
from typing import List

def transform(grid: List[List[int]]) -> List[List[int]]:
    if not grid or not grid[0]:
        return []
    H = len(grid)
    W = len(grid[0])
    output = [row[:] for row in grid]
    visited = [[False] * W for _ in range(H)]
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    
    for r in range(H):
        for c in range(W):
            if grid[r][c] == 8 and not visited[r][c]:
                # New component
                min_row, max_row = r, r
                min_col, max_col = c, c
                q = deque([(r, c)])
                visited[r][c] = True
                while q:
                    x, y = q.popleft()
                    min_row = min(min_row, x)
                    max_row = max(max_row, x)
                    min_col = min(min_col, y)
                    max_col = max(max_col, y)
                    for dx, dy in directions:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < H and 0 <= ny < W and not visited[nx][ny] and grid[nx][ny] == 8:
                            visited[nx][ny] = True
                            q.append((nx, ny))
                # Now fill the bounding box
                for fill_r in range(min_row, max_row + 1):
                    for fill_c in range(min_col, max_col + 1):
                        if grid[fill_r][fill_c] == 0:
                            output[fill_r][fill_c] = 2
    
    return output