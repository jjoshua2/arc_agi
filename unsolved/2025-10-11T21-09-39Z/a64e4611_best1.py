from collections import deque
import copy

def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    if not grid_lst or not grid_lst[0]:
        return []
    rows = len(grid_lst)
    cols = len(grid_lst[0])
    output = copy.deepcopy(grid_lst)
    visited = [[False] * cols for _ in range(rows)]
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for i in range(rows):
        for j in range(cols):
            if grid_lst[i][j] == 0 and not visited[i][j]:
                component = []
                is_border_component = False
                queue = deque([(i, j)])
                visited[i][j] = True

                while queue:
                    x, y = queue.popleft()
                    component.append((x, y))
                    if x == 0 or x == rows - 1 or y == 0 or y == cols - 1:
                        is_border_component = True

                    for dx, dy in directions:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < rows and 0 <= ny < cols and grid_lst[nx][ny] == 0 and not visited[nx][ny]:
                            visited[nx][ny] = True
                            queue.append((nx, ny))

                if not is_border_component:
                    for x, y in component:
                        output[x][y] = 3

    return output