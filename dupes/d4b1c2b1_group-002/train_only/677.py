from collections import deque
from typing import List

def transform(grid: List[List[int]]) -> List[List[int]]:
    if not grid or not grid[0]:
        return []
    rows, cols = len(grid), len(grid[0])
    visited = set()
    q = deque()
    # Add all border 0's to queue
    for i in range(rows):
        # left border
        if grid[i][0] == 0:
            q.append((i, 0))
            visited.add((i, 0))
        # right border
        if grid[i][cols - 1] == 0:
            q.append((i, cols - 1))
            visited.add((i, cols - 1))
    for j in range(cols):
        # top border
        if grid[0][j] == 0:
            q.append((0, j))
            visited.add((0, j))
        # bottom border
        if grid[rows - 1][j] == 0:
            q.append((rows - 1, j))
            visited.add((rows - 1, j))
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    while q:
        x, y = q.popleft()
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < rows and 0 <= ny < cols and (nx, ny) not in visited and grid[nx][ny] == 0:
                visited.add((nx, ny))
                q.append((nx, ny))
    # Create output
    output = [row[:] for row in grid]
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == 0 and (i, j) not in visited:
                output[i][j] = 8
    return output