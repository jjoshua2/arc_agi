from collections import deque
from typing import List

def transform(grid: List[List[int]]) -> List[List[int]]:
    if not grid or not grid[0]:
        return []
    rows = len(grid)
    cols = len(grid[0])
    output = [row[:] for row in grid]
    visited = [[False] * cols for _ in range(rows)]
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    components = []
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == 2 and not visited[i][j]:
                comp = []
                q = deque([(i, j)])
                visited[i][j] = True
                while q:
                    x, y = q.popleft()
                    comp.append((x, y))
                    for dx, dy in directions:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < rows and 0 <= ny < cols and not visited[nx][ny] and grid[nx][ny] == 2:
                            visited[nx][ny] = True
                            q.append((nx, ny))
                components.append(comp)
    if not components:
        return output
    comp_with_avg = []
    for comp in components:
        if comp:
            total_col = sum(c for r, c in comp)
            avg = total_col / len(comp)
            comp_with_avg.append((avg, comp))
    comp_with_avg.sort(key=lambda x: x[0])
    N = len(comp_with_avg)
    parity = N % 2
    for idx in range(N):
        if (idx % 2) != parity:
            avg, comp = comp_with_avg[idx]
            for r, c in comp:
                output[r][c] = 8
    return output