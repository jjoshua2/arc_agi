from collections import deque

def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    if not grid_lst or not grid_lst[0]:
        return []
    rows = len(grid_lst)
    cols = len(grid_lst[0])
    visited = [[False for _ in range(cols)] for _ in range(rows)]
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    count = 0
    for i in range(rows):
        for j in range(cols):
            if grid_lst[i][j] == 0 and not visited[i][j]:
                count += 1
                q = deque([(i, j)])
                visited[i][j] = True
                while q:
                    x, y = q.popleft()
                    for dx, dy in directions:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < rows and 0 <= ny < cols and not visited[nx][ny] and grid_lst[nx][ny] == 0:
                            visited[nx][ny] = True
                            q.append((nx, ny))
    output = [[0] for _ in range(count)]
    return output