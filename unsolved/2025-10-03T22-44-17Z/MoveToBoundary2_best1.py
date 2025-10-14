from collections import deque
from typing import List, Tuple

def is_closed(comp: List[Tuple[int, int]], grid: List[List[int]], rows: int, cols: int) -> bool:
    temp = [[0] * cols for _ in range(rows)]
    for r, c in comp:
        temp[r][c] = 1  # wall
    visited = [[False] * cols for _ in range(rows)]
    q = deque()
    for i in range(rows):
        for j in range(cols):
            if (i == 0 or i == rows - 1 or j == 0 or j == cols - 1) and temp[i][j] == 0 and not visited[i][j]:
                q.append((i, j))
                visited[i][j] = True
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    while q:
        x, y = q.popleft()
        for dx, dy in dirs:
            nx = x + dx
            ny = y + dy
            if 0 <= nx < rows and 0 <= ny < cols and not visited[nx][ny] and temp[nx][ny] == 0:
                visited[nx][ny] = True
                q.append((nx, ny))
    # check for enclosed 0 adjacent to wall
    for r, c in comp:
        for dx, dy in dirs:
            nr = r + dx
            nc = c + dy
            if 0 <= nr < rows and 0 <= nc < cols and temp[nr][nc] == 0 and not visited[nr][nc]:
                return True
    return False

def transform(grid: List[List[int]]) -> List[List[int]]:
    if not grid or not grid[0]:
        return []
    rows = len(grid)
    cols = len(grid[0])
    visited = [[False] * cols for _ in range(rows)]
    components = []
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] != 0 and not visited[i][j]:
                comp = []
                q = deque([(i, j)])
                visited[i][j] = True
                while q:
                    x, y = q.popleft()
                    comp.append((x, y))
                    for dx, dy in dirs:
                        nx = x + dx
                        ny = y + dy
                        if 0 <= nx < rows and 0 <= ny < cols and not visited[nx][ny] and grid[nx][ny] != 0:
                            visited[nx][ny] = True
                            q.append((nx, ny))
                if comp:
                    min_r = min(r for r, _ in comp)
                    max_r = max(r for r, _ in comp)
                    components.append((min_r, max_r, comp))
    # separate closed and open
    closed = []
    openn = []
    for min_r, max_r, comp in components:
        if is_closed(comp, grid, rows, cols):
            closed.append((min_r, max_r, comp))
        else:
            openn.append((min_r, max_r, comp))
    # sort closed by increasing min_r
    closed.sort(key=lambda x: x[0])
    # sort open by decreasing min_r
    openn.sort(key=lambda x: -x[0])
    # output grid
    output = [[0] * cols for _ in range(rows)]
    # place closed components
    for min_r, max_r, comp in closed:
        height = max_r - min_r + 1
        placed = False
        for s in range(rows - height + 1):
            overlap = any(output[s + (r - min_r)][c] != 0 for r, c in comp)
            if not overlap:
                for r, c in comp:
                    nr = s + (r - min_r)
                    output[nr][c] = grid[r][c]
                placed = True
                break
        assert placed
    # place open components
    for min_r, max_r, comp in openn:
        height = max_r - min_r + 1
        placed = False
        for s in range(rows - height, -1, -1):
            overlap = any(output[s + (r - min_r)][c] != 0 for r, c in comp)
            if not overlap:
                for r, c in comp:
                    nr = s + (r - min_r)
                    output[nr][c] = grid[r][c]
                placed = True
                break
        assert placed
    return output